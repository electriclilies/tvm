import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.utils import get_pad_tuple2d
from tvm.relay.dataflow_pattern import rewrite, wildcard, is_op, DFPatternCallback
from collections import OrderedDict
import copy

def _bind_params(func, params): # TODO: make into a util
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.expr.const(v)
    return relay.expr.bind(func, bind_dict)


def prerequisite_optimize(mod, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = tvm.transform.Sequential(
        [relay.transform.SimplifyInference(),
         relay.transform.FoldConstant(),
         relay.transform.FoldScaleAxis(),
         relay.transform.CanonicalizeOps(), #TODO: should this be in prereq optimize?
         relay.transform.FoldConstant()])

    if params is not None:
        mod['main'] = _bind_params(mod['main'], params) # TODO: replace with official bind_params

    with relay.build_config(opt_level=3):
        mod = optimize(mod)
    return mod

# Returns (quantized_mod, calibration_map), where calibration_map maps the scale and zero_point
# vars for a quantized layer to the values before and after quantization. If quantize_all_calibration_layers is true,
# all layers before the current layer will be quantized. Otherwise, the layers before the current layer will be 
# unquantized (ie, the same as the layers in the pre-optimized graph)
# calibration_map is (node_scale, node_zp) = (input, quantized_input, output, quantized_output)
# note that quantized_output may contain an

# TODO: What should default target, ctx be
def quantize(mod, target, ctx, params=None, skip_layers=[0], per_channel=False): #TODO: what should skip layers default be?
    class QuantizeMutator(ExprMutator):
        def __init__(self, inputs, target, ctx, params, skip_layers=[0]):
            super().__init__()

            self.inputs = inputs
            self.target = target
            self.ctx = ctx
            self.params = params

            # if we are skipping layers, remove duplicates and sort
            self.skip_layers = list(set(skip_layers))
            self.skip_layers.sort()
            self.skip_layers_ptr = 0

            # number of conv2d and dense we have seen
            self.compute_layer_count = 0
            
            # counts the number of times we've added a scale and zp for variable naming
            self.scales_count = 0
            self.zp_count = 0
        
            self.calibration_map = OrderedDict()

        # helper to construct the relay scale variable for this layer
        def scale(self, name):
            var = relay.var(str(name) + "_scale_" + str(self.scales_count), shape=(), dtype='float32')

            self.scales_count = self.scales_count + 1
            return var

        # helper to construct the relay zero_point variable for this layer
        def zero_point(self, name):
            var = relay.var(str(name) + "_zero_pt_" + str(self.zp_count), shape=(), dtype='int32')

            self.zp_count = self.zp_count + 1
            return var

        # Helper to wrap the entries in a calibration_map value in relay functions, so they are executable
        # Also compiles the function to a mod, so that it is immediately executable, with the provided
        # target, ctx, and params
        def subgraph_to_mod(self, subgraph):
            return None
            # If inputs are not in the graph, it is constant, so we can just evaluate it now.
            func = relay.Function(list(set(list(self.inputs) + list(relay.analysis.free_vars(subgraph)))), subgraph)
        
            subgraph_mod = tvm.ir.IRModule()
            subgraph_mod['main'] = func

            with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
                lib = relay.build(subgraph_mod, target=self.target)

            module = graph_runtime.GraphModule(lib["default"](self.ctx))
        
            if self.params:
                module.set_input(**self.params)
        
            return module

        def visit_call(self, call):
            if call.op == relay.op.get('nn.conv2d'):
                
                pre_data, pre_weight = call.args[0], call.args[1]
                data, weight = self.visit(pre_data), self.visit(pre_weight)

                # Check if we are skipping quantization on this layer after recursive call
                self.compute_layer_count = self.compute_layer_count + 1 # conv2d is compute layer
                if self.skip_layers and (self.skip_layers_ptr < len(self.skip_layers)): # skip_layers is false if is [] or None
                    if (self.compute_layer_count == self.skip_layers[self.skip_layers_ptr] + 1):
                        self.skip_layers_ptr = self.skip_layers_ptr + 1
                        return super().visit_call(call)

                # Find kernel_size, channels (will be passed to qnn.conv2d)
                new_attr_dict = {}
                kernel_type_info = infer_type(call.args[1])
                for attr in call.attrs.keys():
                    attr_value = call.attrs[attr]
                    if isinstance(attr_value, tvm.ir.container.Array):
                        attr_value = tuple(attr_value)
                    if attr == 'kernel_size':
                        kernel_size = call.attrs[attr]
                        if kernel_size is None:
                            kernel_size = tuple(kernel_type_info.checked_type.shape[2:4]) #assumes OIHW layout
                        else:
                            kernel_size = tuple([k.value for k in call.attrs[attr]])
                    elif attr == 'channels':
                        channels = call.attrs[attr]
                        if channels is None:
                            channels = kernel_type_info.checked_type.shape[0] #TODO: change to work with more layouts
                        channels = channels.value
                    elif attr == 'padding':
                        padding = call.attrs[attr]
                    else:
                        new_attr_dict[str(attr)] = attr_value

                #TODO Figure out if this could be better.
                # Override output dtype.
                new_attr_dict['out_dtype'] = 'int32'

                # Create quantization variables for arguments to this convolution.
                data_scale = self.scale('conv2d_data') 
                data_zp = self.zero_point('conv2d_data')
                weight_scale = self.scale('conv2d_weight')
                weight_zp = self.zero_point('conv2d_weight')
                
                # Quantize the data and construct args for qnn.conv2d
                quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)

                args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale, kernel_size, channels]
                
                if padding is not None:
                    #TODO: Need to make this work with other layouts.
                    top, left, bottom, right = [p.value for p in get_pad_tuple2d(padding)]
                    pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
                    pad_val = 0
                    args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

                # Construct quantized qnn.conv2d and dequantize
                qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
                dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)
                
                pre_data_mod = self.subgraph_to_mod(pre_data)
                quantized_data_mod = self.subgraph_to_mod(quantized_data)
                pre_weight_mod = self.subgraph_to_mod(pre_weight)
                quantized_weight_mod = self.subgraph_to_mod(quantized_weight)
                call_mod = self.subgraph_to_mod(call)
                dequantized_call_mod = self.subgraph_to_mod(dequantized_call)
                
                self.calibration_map[(data_key, weight_key)] = (((pre_data_mod, quantized_data_mod), (pre_weight_mod, quantized_weight_mod)), (call_mod, dequantized_call_mod))
                
                #TODO: fuse bias_add with conv2d during quantization

                return dequantized_call

            if call.op == relay.op.get('nn.dense'):

                pre_data, pre_weight = call.args[0], call.args[1]
                data, weight = self.visit(pre_data), self.visit(pre_weight)

                # Check if we are skipping quantization on this layer after recursive call
                self.compute_layer_count = self.compute_layer_count + 1 # dense is a compute layer
                if self.skip_layers and (self.skip_layers_ptr < len(self.skip_layers)):
                    if (self.compute_layer_count == self.skip_layers[self.skip_layers_ptr] + 1):
                        self.skip_layers_ptr = self.skip_layers_ptr + 1
                        return super().visit_call(call)

                # Create quantization parameters for arguments to this dense layer.
                data_scale = self.scale('dense_data') 
                data_zp = self.zero_point('dense_data')
                weight_scale = self.scale('dense_weight')
                weight_zp = self.zero_point('dense_weight')
                
                # Quantize data and construct args for qnn.dense
                quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)

                units = call.attrs['units']
                if units is None:
                    weight_type_info = infer_type(call.args[1])
                    units = weight_type_info.checked_type.shape[0]
                
                args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale, units]

                qnn_call = relay.qnn.op.dense(*args)
                dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)

                pre_data_mod = self.subgraph_to_mod(pre_data)
                quantized_data_mod = self.subgraph_to_mod(quantized_data)
                pre_weight_mod = self.subgraph_to_mod(pre_weight)
                quantized_weight_mod = self.subgraph_to_mod(quantized_weight)
                call_mod = self.subgraph_to_mod(call)
                dequantized_call_mod = self.subgraph_to_mod(dequantized_call)

                self.calibration_map[(data_key, weight_key)] = (((pre_data_mod, quantized_data_mod), (pre_weight_mod, quantized_weight_mod)), (call_mod, dequantized_call_mod))

                return dequantized_call

            if call.op == relay.op.get('add'):
                pre_lhs, pre_rhs = call.args[0], call.args[1]
                lhs, rhs = self.visit(pre_lhs), self.visit(pre_rhs)
                
                # Don't quantize the add if it is in a skipped layer
                if self.skip_layers and self.skip_layers_ptr != 0:
                    last_layer = self.compute_layer_count - 1
                    last_skipped = self.skip_layers[self.skip_layers_ptr - 1]
                    if (last_layer == last_skipped):
                        return super().visit_call(call)
                
                 # Create quantization parameters for arguments to this addition
                lhs_scale = self.scale('add_lhs') 
                lhs_zp = self.zero_point('add_lhs')
                rhs_scale = self.scale('add_rhs')
                rhs_zp = self.zero_point('add_rhs')

                # Quantize, dequantize, and requantize inputs to have scale lhs_scale + rhs_scale
                # (Scale represents the lowest possible value representable in the quantized type,
                # so the smallest representable output is lhs_scale + rhs_scale)
                
                # We do this to avoid the requantize op in qnn's add, which causes issues with compilation
                # Requantize will be inserted in a future pass
                quantized_lhs = relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp)
                quantized_rhs = relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)

                dequantized_lhs = relay.qnn.op.dequantize(quantized_lhs, lhs_scale, relay.const(0, dtype='int32'))
                dequantized_rhs = relay.qnn.op.dequantize(quantized_rhs, rhs_scale, relay.const(0, dtype='int32'))

                add_scale = relay.op.add(lhs_scale, rhs_scale)

                requantized_lhs = relay.qnn.op.quantize(dequantized_lhs, add_scale, relay.const(0, dtype='int32'))
                requantized_rhs = relay.qnn.op.quantize(dequantized_rhs, add_scale, relay.const(0, dtype='int32'))
        
                add = relay.op.add(requantized_lhs, requantized_rhs)
                dequantized_call = relay.qnn.op.dequantize(add, add_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                lhs_key = (lhs_scale, lhs_zp)
                rhs_key = (rhs_scale, rhs_zp)
                
                pre_lhs_mod = self.subgraph_to_mod(pre_lhs)
                quantized_lhs_mod = self.subgraph_to_mod(quantized_lhs)
                pre_rhs_mod = self.subgraph_to_mod(pre_rhs)
                quantized_rhs_mod = self.subgraph_to_mod(quantized_rhs)
                call_mod = self.subgraph_to_mod(call)
                dequantized_call_mod = self.subgraph_to_mod(dequantized_call)

                self.calibration_map[(lhs_key, rhs_key)] = (((pre_lhs_mod, quantized_lhs_mod), (pre_rhs_mod, quantized_rhs_mod)), (call_mod, dequantized_call_mod))

                return dequantized_call

            if call.op == relay.op.get('multiply'):
                pre_lhs, pre_rhs = call.args[0], call.args[1]
                lhs, rhs = self.visit(pre_lhs), self.visit(pre_rhs)

                # Don't quantize multiply if it is in a skipped layer
                if self.skip_layers and self.skip_layers_ptr != 0:
                    last_layer = self.compute_layer_count - 1
                    last_skipped = self.skip_layers[self.skip_layers_ptr - 1]
                    if (last_layer == last_skipped):
                        return super().visit_call(call)

                # Create quantization parameters for arguments to this multiplication.
                lhs_scale = self.scale('mul_lhs')
                lhs_zp = self.zero_point('mul_lhs')
                rhs_scale = self.scale('mul_rhs')
                rhs_zp = self.zero_point('mul_rhs')

                # Quantize inputs and construct args for multiply
                quantized_lhs = tvm.relay.cast(relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp), 'int32')
                quantized_rhs = tvm.relay.cast(relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp), 'int32')

                # Use normal relay multiply instead of qnn multiply to avoid requantize in qnn.mul
                # Subtract zero points to center on zero so that we can multiply lhs, rhs directly
                zeroed_quantized_lhs = relay.op.subtract(quantized_lhs, lhs_zp)
                zeroed_quantized_rhs = relay.op.subtract(quantized_rhs, rhs_zp)
                
                multiply = relay.op.multiply(zeroed_quantized_lhs, zeroed_quantized_rhs)
                dequantized_call = relay.qnn.op.dequantize(multiply, lhs_scale * rhs_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                lhs_key = (lhs_scale, lhs_zp)
                rhs_key = (rhs_scale, rhs_zp)

                pre_lhs_mod = self.subgraph_to_mod(pre_lhs)
                quantized_lhs_mod = self.subgraph_to_mod(quantized_lhs)
                pre_rhs_mod = self.subgraph_to_mod(pre_rhs)
                quantized_rhs_mod = self.subgraph_to_mod(quantized_rhs)
                call_mod = self.subgraph_to_mod(call)
                dequantized_call_mod = self.subgraph_to_mod(dequantized_call)

                self.calibration_map[(lhs_key, rhs_key)] = (((pre_lhs_mod, quantized_lhs_mod), (pre_rhs_mod, quantized_rhs_mod)), (call_mod, dequantized_call_mod))

                return dequantized_call
            
            else:
                return super().visit_call(call)

    # SimplifyInference, FoldConstants, FoldScaleAxis
    preoptimized_mod = prerequisite_optimize(mod, params)
    inputs = preoptimized_mod['main'].params
    quantize_pass = QuantizeMutator(inputs, target, ctx, params, skip_layers)

    q_fn = quantize_pass.visit(preoptimized_mod['main'])
    q_fn = relay.Function(list(q_fn.params) + list(relay.analysis.free_vars(q_fn)), q_fn.body)
    
    quantized_mod = tvm.ir.IRModule()
    quantized_mod['main'] = q_fn
    # we return a mod for consistency with other passes
    return (quantized_mod, quantize_pass.calibration_map)


# Optionally repeats pattern n times. If n = 1, returns pattern. Otherwise, returns a pattern that will match
# pattern repeated any number of times up to n
def repeat_pattern(pattern, n):
    for _ in range(n):
        pattern = pattern.optional(lambda x: pattern(x))
    return pattern

def requantize(mod):
    class RequantizeCallback(DFPatternCallback):
        def __init__(self):
            super(RequantizeCallback, self).__init__()
            
            #this matches the entire graph. oops. 
            """
            base_case = is_op("add")(wildcard(), wildcard()) | is_op("multiply")(wildcard(), wildcard()) | is_op("nn.max_pool2d")(wildcard())
            is_supported_op = is_op("add")(base_case, wildcard()) | is_op("add")(wildcard(), base_case) | is_op("multiply")(base_case, wildcard()) | is_op("multiply")(wildcard(), base_case) | is_op("nn.max_pool2d")(base_case)

            rec_depth = 10
            is_supported_op_repeat = repeat_pattern(is_supported_op, rec_depth)
            
            self.pattern = is_op("nn.relu")(is_supported_op_repeat) # it looks like this can result in an infinite loop within the pattern matcher so BE CAREFUL!!! is_op("nn.relu").optional(lambda x : is_supported_op(x))
            """
            is_supported_op = is_op("add")(wildcard(), wildcard()) | is_op("multiply")(wildcard(), wildcard()) | is_op("nn.max_pool2d")(wildcard())
            self.pattern = repeat_pattern(is_supported_op, 20)


        def callback(self, pre, post, node_map):
            print("saw pattern")
            print("pre: ", pre)
            return None
            """
            dequantize_data = node_map[self.dequantize_data][0]

            input_scale = node_map[self.input_scale][0]
            input_zero_point = node_map[self.input_zero_point][0]
            output_scale = node_map[self.output_scale][0]
            output_zero_point = node_map[self.output_zero_point][0]

            first_add_arg = node_map[self.first_add_arg][0]

            # optional add
            if self.second_add_arg in node_map:
                second_add_arg = node_map[self.second_add_arg][0]
            else:
                second_add_arg = None

            # optional maxpool
            # how do I tell if maxpool was in the pattern?

            # TODO: requantize before or after ops?
            requantize = relay.qnn.op.requantize(dequantize_data, input_scale, input_zero_point, output_scale, output_zero_point)

            add = relay.op.add(requantize, relay.qnn.op.quantize(first_add_arg, output_scale, output_zero_point))
            # if 2 adds in a row, construct the second
            if second_add_arg:
                add = relay.op.add(add, relay.qnn.op.quantize(second_add_arg, output_scale, output_zero_point))
            relu = relay.op.nn.relu(add)
            # do we want to requantize before or after the ops
            return relu
            """
    return rewrite(RequantizeCallback(), mod)
