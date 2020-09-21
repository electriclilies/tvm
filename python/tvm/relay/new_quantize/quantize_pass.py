import tvm
from tvm import relay
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type


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
                zero_point = relay.const(0, dtype='int32')
                scale = relay.const(1, dtype='float32')
                args = args + [zero_point, zero_point, scale, scale]
                new_attr_dict= {}
                for attr in call.attrs.keys():
                    attr_value = call.attrs[attr]
                    if isinstance(attr_value, tvm.ir.container.Array):
                        attr_value = tuple(attr_value)
                    if attr == 'kernel_size':
                        kernel_size = call.attrs[attr]
                        if kernel_size is None:
                            type_info = infer_type()
                            kernel_size = tuple(type_info.checked_type.shape[2:4]) #assumes OIHW layout
                        else:
                            kernel_size = tuple([k.value for k in call.attrs[attr]])
                    elif attr == 'channels':
                        channels = call.attrs[attr]
                        if channels is None:
                            type_info = infer_type(args[1])
                            channels = type_info.checked_type.shape[0] #TODO: change to work with more layouts
                        channels = channels.value
                    else:
                        new_attr_dict[str(attr)] = attr_value
                args = args + [kernel_size, channels]
                #TODO Figure out if this could be better.
                # Override output dtype.
                new_attr_dict['out_dtype'] = 'int32'
                qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
                return relay.qnn.op.dequantize(qnn_call, scale, zero_point)
            else:
                return super().visit_call(call)

    # SimplifyInference, FoldConstants, FoldScaleAxis
    mod = prerequisite_optimize(mod, params)
    quantize_pass = QuantizeMutator()
    mod['main'] = quantize_pass.visit(mod['main'])
    return mod