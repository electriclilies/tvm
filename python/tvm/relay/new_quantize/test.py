import tvm
import onnx
import torch
from torchvision.models import resnet
from tvm import relay
from tvm.relay.new_quantize import Quantizer, GlobalCalibrater, Requantizer
from tvm.contrib import graph_runtime
import numpy as np
import tvm.testing


batch_size = 5
# Import onnx model, quantize and calibrate
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/mnist_model.onnx')
input_dict = {'flatten_input': [batch_size, 28, 28, 1]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
#print("Unquantized mod: ")
#print(mod.astext(False))
#print(" ____________________________ ")

quantized_mod, calibration_map = Quantizer().quantize(mod, params, skip_layers=[])
#print("Quantized mod: ")
#print(quantized_mod.astext(False))

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    q_lib = relay.build(quantized_mod, target='llvm')

gc = GlobalCalibrater(0.05, 0)
calibrated_mod = gc.calibrate(quantized_mod, calibration_map)

rq = Requantizer()
print("Requantizing..")
requantized_mod = rq.requantize(calibrated_mod)
print(requantized_mod.astext(False))

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    q_calibrated_lib = relay.build(requantized_mod, target='llvm')
print("Built")
input_np = np.random.randn(batch_size, 28, 28, 1).astype('float32')

gmod = graph_runtime.GraphModule(q_calibrated_lib["default"](tvm.cpu()))

gmod.set_input(**{'flatten_input': input_np})
gmod.run()
out = gmod.get_output(0).asnumpy()
print(out)
print("Unquantized Output:")
print("Build calibrated mod successfully")
print("Small MNIST model worked")

pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shape = (1, 3, 224, 224)
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)
mod = relay.transform.InferType()(mod)
print("Resnet 18 mod: ", mod)
quantized_mod, calibration_map = Quantizer().quantize(mod, params, skip_layers=[])
print("successfully quantized")
# For testing purposes, manually set scale and zp to bad values

input_np = np.random.randn(1, 3, 224, 224).astype('float32')
print("calibrating")
global_calibrater = GlobalCalibrater(0.05, 0)
calibrated_mod = global_calibrater.calibrate(quantized_mod, calibration_map)
print("calibrated")
print(calibrated_mod)
print("Requantizing")
requantized_mod = rq.requantize(calibrated_mod)
print(requantized_mod.astext(False))

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(requantized_mod, target='llvm')

from tvm.contrib import graph_runtime