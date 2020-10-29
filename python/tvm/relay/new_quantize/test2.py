import tvm
from tvm import relay
from tvm.relay.quantize import _find_scale_by_kl

import numpy as np

np.random.seed(0)
shape = (1, )
input_np = np.random.uniform(size=shape)
scale_value = _find_scale_by_kl(input_np)

data = relay.var("data", shape=shape, dtype="float32")
q_data = relay.qnn.op.quantize(data, relay.const(scale_value, dtype='float32'), relay.const(0, dtype='int32'))
deq_data = relay.qnn.op.dequantize(q_data, relay.const(scale_value, dtype='float32'), relay.const(0, dtype='int32'))

ir_mod = tvm.ir.IRModule()
ir_mod["main"] = relay.Function([data], deq_data)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(ir_mod, target='llvm')

from tvm.contrib import graph_runtime
gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
gmod.set_input(**{'data': input_np})
gmod.run()
out = gmod.get_output(0).asnumpy()

print("Original array: ")
print(input_np)
print("Quantized array with scale ", scale_value)
print(out)
