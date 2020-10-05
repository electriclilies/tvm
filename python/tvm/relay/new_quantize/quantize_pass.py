import tvm
from tvm import relay
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.util import get_pad_tuple2d
from tvm.relay.dataflow_pattern import rewrite, wildcard, is_op, DFPatternCallback

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
         relay.transform.CanonicalizeOps(),
         relay.transform.FoldConstant()])

    if params is not None:
        mod['main'] = _bind_params(mod['main'], params)

    with relay.build_config(opt_level=3):
        mod = optimize(mod)
    return mod

# Returns (preoptimized_funcs, quantized_func, calibration_map), where calibration_map maps the scale and zero_point
# vars for a quantized layer to the values before and after quantization. If quantize_all_calibration_layers is true,
# all layers before the current layer will be quantized. Otherwise, the layers before the current layer will be 
# unquantized (ie, the same as the layers in the pre-optimized graph)
# Key is always (node_scale, node_zp)
# TODO: find a better name for quantize_all_calibration_layers

# TODO: should we use relay vars themselves as keys or strings? probably vars.
def quantize(mod, params, skip_layers=[], quantize_all_calibration_layers=True):
    class QuantizeMutator(ExprMutator):
        def __init__(self, skip_layers):
            super().__init__()
            
            # if we are skipping layers, remove duplicates and sort
            self.skip_layers = list(set(skip_layers))
            self.skip_layers.sort()
            self.skip_layers_ptr = 0

            # number of conv2d and dense we have seen
            self.compute_layer_count = 0
            
            self.calibration_map = {} #TODO: make into its own class

            self.scales = [] #TODO: connect me to node_map somehow, also maybe change from set bc bad runtime
            self.zero_points = []
        
        # helper to construct the relay scale variable for this layer
        def scale(self, name):
            var = relay.var(str(name) + "_scale_" + str(len(self.scales)))
            self.scales.append(var)
            return var

        # helper to construct the relay zero_point variable for this layer
        def zero_point(self, name):
            var = relay.var(str(name) + "_zero_pt_" + str(len(self.zero_points)))
            self.zero_points.append(var)
            return var

        # helper to construct dictionary of output for before and after quantization
        # TODO : is dictionary the best structure for this? would tuple be better?
        def calibration_map_helper(self, before_quantize, after_quantize):
            return {'before_quantize': before_quantize, 'after_quantize' : after_quantize}

        def visit_call(self, call):
            if call.op == relay.op.get('nn.conv2d'):
                data, weight = self.visit(call.args[0]), self.visit(call.args[1])
                # check if we are skipping quantization on this layer after recursive call
                self.compute_layer_count = self.compute_layer_count + 1 # conv2d is compute layer
                if self.skip_layers and (self.skip_layers_ptr < len(self.skip_layers)): # skip_layers is false if is [] or None
                    print("skip_layers_ptr", self.skip_layers_ptr)
                    if (self.compute_layer_count == self.skip_layers[self.skip_layers_ptr] + 1):
                        self.skip_layers_ptr = self.skip_layers_ptr + 1
                        return super().visit_call(call)

                # Create quantization parameters for arguments to this convolution.
                data_scale = self.scale('conv2d_data') 
                data_zp = self.zero_point('conv2d_data')
                weight_scale = self.scale('conv2d_weight')
                weight_zp = self.zero_point('conv2d_weight')

                quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)
                
                args = [quantized_data, quantized_weight]

                # put scale and zp in calibration_map
                # TODO: is this the best way to do KEY?
                # TODO: does this read better if I consolidate it all at the end of the function? 
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)
                
                if quantize_all_calibration_layers:
                    self.calibration_map[data_key] = self.calibration_map_helper(data, quantized_data)
                    self.calibration_map[weight_key] = self.calibration_map_helper(weight, quantized_weight)
                else:
                    # pre_data and pre_weight are the data and weight before we quantized them 
                    pre_data, pre_weight = (call.args[0], call.args[1])
                    quantized_data = relay.qnn.op.quantize(pre_data, data_scale, data_zp)
                    quantized_weight = relay.qnn.op.quantize(pre_weight, weight_scale, weight_zp)
                    self.calibration_map[data_key] = self.calibration_map_helper(pre_data, quantized_data)
                    self.calibration_map[weight_key] = self.calibration_map_helper(pre_weight, quantized_weight)

                args = args + [data_zp, weight_zp, data_scale, weight_scale]

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
                args = args + [kernel_size, channels]
                #TODO Figure out if this could be better.
                # Override output dtype.
                new_attr_dict['out_dtype'] = 'int32'

                if padding is not None:
                    #TODO Need to make this work with other layouts.
                    top, left, bottom, right = [p.value for p in get_pad_tuple2d(padding)]
                    pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
                    pad_val = 0
                    args[0] = relay.op.nn.pad(args[0], pad_width, pad_val) 

                qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
                out = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'))
                # TODO: Do we need to do any comparison between the dequantized and the unquantized output? If so, need another scheme for multiplied stuff..
                # in some sense, the output of this is what we really care about though... 

                return out

            if call.op == relay.op.get('nn.dense'):
                data, weight = self.visit(call.args[0]), self.visit(call.args[1])
                # check if we are skipping quantization on this layer after recursive call
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
                
                args = [relay.qnn.op.quantize(data, data_scale, data_zp),
                        relay.qnn.op.quantize(weight, weight_scale, weight_zp)]
                                

                args = args + [data_zp, weight_zp, data_scale, weight_scale, call.attrs['units']]

                qnn_call = relay.qnn.op.dense(*args)
                out = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'))
                
                self.node_map[out] = call
                return out

            if call.op == relay.op.get('add'):
                lhs = self.visit(call.args[0])
                rhs = self.visit(call.args[1])
                # don't quantize add if it follows a skipped layer
                if self.skip_layers and self.skip_layers_ptr is not 0:
                    last_layer = self.compute_layer_count - 1
                    last_skipped = self.skip_layers[self.skip_layers_ptr - 1]
                    if (last_layer == last_skipped):
                        return super().visit_call(call)

                 # Create quantization parameters for arguments to this addition
                lhs_scale = self.scale('add_lhs') 
                lhs_zp = self.zero_point('add_lhs')
                rhs_scale = self.scale('add_rhs')
                rhs_zp = self.zero_point('add_rhs')
                out_scale = self.scale('add_out')
                out_zp = self.zero_point('add_out')

                args = [relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp),
                        relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)]

                args = args + [lhs_scale, lhs_zp, rhs_scale, rhs_zp, out_scale, out_zp]
                q_add = relay.qnn.op.add(*args)
                out = relay.qnn.op.dequantize(q_add, out_scale, out_zp)

                self.node_map[out] = call
                return out

            if call.op == relay.op.get('multiply'):
                lhs = self.visit(call.args[0])
                rhs = self.visit(call.args[1])
                # don't quantize add if it follows a skipped layer
                if self.skip_layers and self.skip_layers_ptr is not 0:
                    last_layer = self.compute_layer_count - 1
                    last_skipped = self.skip_layers[self.skip_layers_ptr - 1]
                    if (last_layer == last_skipped):
                        return super().visit_call(call)

                # Create quantization parameters for arguments to this multiplication.
                lhs_scale = self.scale('mul_lhs')
                lhs_zp = self.zero_point('mul_lhs')
                rhs_scale = self.scale('mul_rhs')
                rhs_zp = self.zero_point('mul_rhs')
                out_scale = lhs_scale * rhs_scale
                out_zp = relay.const('0', dtype='int32')

                args = [relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp),
                        relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)]
                
                args = args + [lhs_scale, lhs_zp, rhs_scale, rhs_zp, out_scale, out_zp]
                q_mul = relay.qnn.op.add(*args)
                out = relay.qnn.op.dequantize(q_mul, out_scale, out_zp)
                
                self.node_map[out] = call
                return out
            
            else:
                return super().visit_call(call)


    # SimplifyInference, FoldConstants, FoldScaleAxis
    preoptimized_mod = prerequisite_optimize(mod, params)
    quantize_pass = QuantizeMutator(skip_layers)

    q_fn = quantize_pass.visit(preoptimized_mod['main'])
    q_fn = relay.Function(list(q_fn.params) + list(quantize_pass.scales) + list(quantize_pass.zero_points), q_fn.body)

    #mod['main'] = requantize(mod['main'])
    return (preoptimized_mod['main'], q_fn, quantize_pass.node_map)

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
