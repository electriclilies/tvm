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


# Returns (preoptimized_funcs, quantized_func, node_map), where node_map maps each AST node in the preoptimized function
# to the 

def quantize(mod, params, skip_layers=[]):
    class QuantizeMutator(ExprMutator):
        def __init__(self, skip_layers):
            super().__init__()
            self.zero_point = relay.const(0, dtype='int32')
            self.scale = relay.const(1, dtype='float32')
            
            # if we are skipping layers, remove duplicates and sort
            self.skip_layers = list(set(skip_layers))
            self.skip_layers.sort()

            # number of conv2d and dense we have seen
            self.compute_layer_count = 0
            
            self.node_map = {} #TODO: make into its own class

        def visit_call(self, call):
            if call.op == relay.op.get('nn.conv2d'):

                args = [relay.qnn.op.quantize(self.visit(arg), self.scale, self.zero_point) for arg in call.args]

                # check if we are skipping quantization on this layer after recursive call
                if self.skip_layers: # skip_layers is false if is [] or None
                    if (self.compute_layer_count == self.skip_layers[0]):
                        self.skip_layers.pop(0)
                        self.compute_layer_count = self.compute_layer_count + 1
                        return super().visit_call(call)

                self.compute_layer_count = self.compute_layer_count + 1 # conv2d is compute layer

                args = args + [self.zero_point, self.zero_point, self.scale, self.scale]
                kernel_type_info = infer_type(args[1])

                new_attr_dict = {}
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
                out = relay.qnn.op.dequantize(qnn_call, self.scale, self.zero_point)
                
                self.node_map[call] = out
                return out
            if call.op == relay.op.get('nn.dense'):
                
                args = [relay.qnn.op.quantize(self.visit(arg), self.scale, self.zero_point) for arg in call.args]
                
                # check if we are skipping quantization on this layer after recursive call
                if self.skip_layers: # skip_layers is false if is [] or None
                    if (self.compute_layer_count == self.skip_layers[0]):
                        self.skip_layers.pop(0)
                        self.compute_layer_count = self.compute_layer_count + 1
                        return super().visit_call(call)
                
                self.compute_layer_count = self.compute_layer_count + 1 # dense is a compute layer

                args = args + [self.zero_point, self.zero_point, self.scale, self.scale, call.attrs['units']]

                qnn_call = relay.qnn.op.dense(*args)
                out = relay.qnn.op.dequantize(qnn_call, self.scale, self.zero_point)
                
                self.node_map[call] = out
                return out
            if call.op == relay.op.get('add'):
                args = [relay.qnn.op.quantize(self.visit(arg), self.scale, self.zero_point) for arg in call.args]

                q_add = relay.op.add(*args)
                out = relay.qnn.op.dequantize(q_add, self.scale, self.zero_point)

                self.node_map[call] = out
                return out

            if call.op == relay.op.get('multiply'):
                args = [relay.qnn.op.quantize(self.visit(arg), self.scale, self.zero_point) for arg in call.args]

                q_mul = relay.op.add(*args)

                return relay.qnn.op.dequantize(q_mul, self.scale, self.zero_point)
            
            else:
                return super().visit_call(call)


    # SimplifyInference, FoldConstants, FoldScaleAxis
    mod = prerequisite_optimize(mod, params)
    quantize_pass = QuantizeMutator(skip_layers)
    mod['main'] = quantize_pass.visit(mod['main'])

    #mod['main'] = requantize(mod['main'])
    return (mod, quantize_pass.node_map)

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
