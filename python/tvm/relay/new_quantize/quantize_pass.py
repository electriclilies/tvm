import tvm
from tvm import relay
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type

def quantize(func):
    class QuantizeMutator(ExprMutator):
        def visit_call(self, call):
            if call.op == relay.op.get('nn.conv2d'):
                #TODO Actually use quantize instead of hard casting
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
                        #TODO Add handlng for None
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
                #TODO We should dequantize instead of cast.
                return relay.qnn.op.dequantize(qnn_call, scale, zero_point)
            else:
                return super().visit_call(call)

    quantize_pass = QuantizeMutator()
    func = quantize_pass.visit(func)
    return tvm.IRModule.from_expr(func)