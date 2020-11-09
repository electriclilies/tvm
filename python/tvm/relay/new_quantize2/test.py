import _quantizer
import tvm
import onnx
import torch
from torchvision.models import resnet
from tvm import relay
import numpy as np
import tvm.testing


batch_size = 5
# Import onnx model, quantize and calibrate
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/mnist_model.onnx')
input_dict = {'flatten_input': [batch_size, 28, 28, 1]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
print("Unquantized mod: ")
print(mod.astext(False))
print(" ____________________________ ")

quantizer = _quantizer.Quantizer()
quantized_mod, calibration_map = quantizer.quantize(mod)
print("Quantized mod: ")
print(quantized_mod.astext(False))


with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    q_lib = relay.build(calibration_map.q_tuple_subgraph_mod, target='llvm')
    lib = relay.build(calibration_map.tuple_subgraph_mod, target='llvm')

print("Small MNIST model worked")

pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shape = (1, 3, 224, 224)
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)
print("Resnet 18 mod: ", mod)
quantizer = _quantizer.Quantizer()
quantized_mod, calibration_map = quantizer.quantize(mod)

mod = calibration_map.tuple_subgraph_mod
q_mod = calibration_map.q_tuple_subgraph_mod
print(q_mod)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(quantized_mod, target='llvm')




exit()
input_np = np.random.randn(1, 3, 224, 224).astype('float32')

global_calibrater = GlobalCalibrater(0.05, 0, 0.005, 0)
calibrated_mod = global_calibrater.calibrate(quantized_mod, calibration_map, params)

print(calibrated_mod)
#calibrated_mod = global_calibrater.calibrate(quantized_mod, calibration_map, params)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(calibrated_mod, target='llvm')

from tvm.contrib import graph_runtime