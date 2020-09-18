import tvm
import onnx
from tvm import relay
from quantize_pass import quantize

onnx_model = onnx.load('resnet18-v1-7.onnx')
input_dict = {'data': [1, 3, 224, 224]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
mod = quantize(mod['main'])

print(mod['main'])