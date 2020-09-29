import tvm
import onnx
import torch
from torchvision.models import resnet
from tvm import relay
import quantize_pass

# ONNX TEST
"""
onnx_model = onnx.load('resnet18-v1-7.onnx')
input_dict = {'data': [1, 3, 224, 224]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
mod = quantize_pass.quantize(mod, params)

print("ONNX resnet 18 quantized with our quantize pass:")
print(mod['main'])
"""

pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shape = (1, 3, 224, 224)
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)
mod, node_map = quantize_pass.quantize(mod, params, skip_layers=[1, 7, 9, 20])
print(mod)
# print("Pytorch resnet 18 quantized with our quantize pass:")
# print(mod['main'])