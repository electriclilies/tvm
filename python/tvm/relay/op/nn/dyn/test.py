from tvm.relay.op.nn.dyn._nn import _upsampling_nchw_shape_func
import tvm

shape = tvm.te.placeholder((4, 5, 6, 7), 'int64')
sh = tvm.te.placeholder((1, ), 'float32')
sw = tvm.te.placeholder((1, ), 'float32')

scaled_shape = _upsampling_nchw_shape_func(shape, sh, sw)

sch = tvm.te.create_schedule(scaled_shape.op)
ir = tvm.lower(sch, [shape, sh, sw], simple_mode=True)
print("IR: ")
print(ir)

module = tvm.build(sch, [shape, sh, sw], 'llvm')

print("Module source: ")
print(module.get_source())