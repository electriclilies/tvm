import tvm
from tvm import relay
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.util import get_pad_tuple2d
from tvm.relay.dataflow_pattern import rewrite, wildcard, is_op, DFPatternCallback

def _bind_params(func, params):
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


def quantize(mod, params):
    class QuantizeMutator(ExprMutator):
        def visit_call(self, call):
            # do zero_point, scale need to be different for different calls? 
            # can we match dequantize relu maxpool quantize
            # what to do about add? 
            zero_point = relay.const(0, dtype='int32')
            scale = relay.const(1, dtype='float32')
            if call.op == relay.op.get('nn.conv2d'):

                args = [relay.qnn.op.quantize(self.visit(arg), scale, zero_point) for arg in call.args]
                
                args = args + [zero_point, zero_point, scale, scale]
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
                return relay.qnn.op.dequantize(qnn_call, scale, zero_point)

            
            # if call.op == relay.op.get('add'):
            #     # if both args are either dequantize(quantize?) or QNN ops
            #     args = [relay.qnn.op.quantize(self.visit(arg), scale, zero_point) for arg in call.args]

            #     # TODO: change scale, zeropt to be better.. 
            #     new_attrs = {'lhs_scale' : scale, 'lhs_zero_point' : zero_point, 'rhs_scale' : scale,
            #     'rhs_zero_point' : zero_point, 'output_scale' : scale, 'output_zero_point' : zero_point}
            #     q_add = relay.qnn.op.add(*args, **new_attrs)
                
            #     return relay.qnn.op.dequantize(q_add, scale, zero_point)
            else:
                return super().visit_call(call)

    # SimplifyInference, FoldConstants, FoldScaleAxis
    mod = prerequisite_optimize(mod, params)
    quantize_pass = QuantizeMutator()
    mod['main'] = quantize_pass.visit(mod['main'])
    print("prequantized stuff")
    print(mod)
    mod['main'] = requantize(mod['main'])
    return mod

def requantize(mod):
    class RequantizeCallback(DFPatternCallback):
        def __init__(self):
            super(RequantizeCallback, self).__init__()

            self.dequantize_data = wildcard()

            self.input_scale = wildcard()
            self.input_zero_point = wildcard()

            self.output_scale = wildcard()
            self.output_zero_point = wildcard()

            self.first_add_arg = wildcard()
            self.second_add_arg = wildcard()

            is_dequantize_op = is_op('qnn.dequantize')(self.dequantize_data, self.output_scale,
                                      self.output_zero_point)

            # how to copy over the attributes?
            is_add_op = is_op('add')(is_dequantize_op, self.first_add_arg).optional(lambda x: is_op("add")(x, self.second_add_arg))
            #is_maxpool_op = is_add_op.optional(lambda x: is_op("nn.max_pool2d")(is_add_op)) #finish me later
            is_relu_op = is_op('nn.relu')(is_add_op)
            is_quantize_op = is_op('qnn.quantize')(is_relu_op, self.input_scale,
                                 self.input_zero_point)
            self.pattern = is_quantize_op

        def callback(self, pre, post, node_map):
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

            add = relay.op.add(requantize, relay.qnn.op.quantize(first_add_arg, output_scale, output_zero_point)) # TODO: do something better than just casting
            # if 2 adds in a row, construct the second
            if second_add_arg:
                add = relay.op.add(add, relay.qnn.op.quantize(second_add_arg, output_scale, output_zero_point))
            relu = relay.op.nn.relu(add)
            # do we want to requantize before or after the ops
            return relu
    return rewrite(RequantizeCallback(), mod)
    