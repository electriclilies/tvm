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


def quantize(mod, params=None):
    class QuantizeMutator(ExprMutator):
        # add pad explicitly
        def visit_call(self, call):
            if call.op == relay.op.get('nn.conv2d'):

                zero_point = relay.const(0, dtype='int32')
                scale = relay.const(1, dtype='float32')

                args = [relay.qnn.op.quantize(self.visit(arg), scale, zero_point) for arg in call.args]
                
                args = args + [zero_point, zero_point, scale, scale] #is this right now? not sure. 
                kernel_type_info = infer_type(args[1]) # make sure this is right

                new_attr_dict= {}
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
            else:
                return super().visit_call(call)

    # SimplifyInference, FoldConstants, FoldScaleAxis
    mod = prerequisite_optimize(mod, params)
    quantize_pass = QuantizeMutator()
    mod['main'] = quantize_pass.visit(mod['main'])
    mod['main'] = requantize(mod['main'])
    return mod

def requantize(mod):
    class RequantizeCallback(DFPatternCallback):
        def __init__(self):
            self.quantize_data = wildcard()

            self.input_scale = wildcard()
            self.input_zero_point = wildcard()

            self.output_scale = wildcard()
            self.output_zero_point = wildcard()
            
            is_quantize_op = is_op('qnn.quantize')(self.quantize_data, self.input_scale,
                                 self.input_zero_point, wildcard())
            is_dequantize_op = is_op('qnn.dequantize')(wildcard(), self.output_scale,
                                      self.output_zero_point, wildcard())
            
            self.pattern = is_quantize_op(is_dequantize_op)

        def callback(self, pre, post, node_map):
            quantize_data = node_map[self.quantize_data][0]

            input_scale = node_map[self.input_scale][0]
            input_zero_point = node_map[self.input_scale][0]

            output_scale = node_map[self.output_scale][0]
            output_zero_point = node_map[self.output_zero_point][0]
            # do we need to move axis, dtype? (they are attrs -- also how to extract?)
            # what is requantize rounding param do?
            return relay.qnn.op.requantize(quantize_data, input_scale, input_zero_point, output_scale, output_zero_point)

    return rewrite(RequantizeCallback(), mod)
    