
import tvm
import tvm.relay.testing
from tvm import relay
import torch

from torchvision.models import resnet
from tvm.relay.new_quantize import Quantizer, Calibrater, DatasetManager, AverageMaxCalibrationCallback, AverageMaxPerChannelConv2DBiasAddPattern, AverageMaxPerChannelConv2DPattern, AverageMaxPerChannelDensePattern, Requantizer

import numpy as np

class RandomDatasetManager(DatasetManager):
    def __init__(self, data_shape, dtype, num_batches):
        self.idx = 0
        self.data_shape = data_shape
        self.dtype = dtype
        self.n_batches = num_batches
    
    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1
        return [np.random.randn(*self.data_shape).astype(self.dtype)], [None]

    def num_batches(self):
        return self.n_batches

    def is_empty(self):
        return self.idx >= self.n_batches

    def reset(self):
        self.idx = 0


pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shape = (3, 3, 224, 224)
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)

cc = AverageMaxCalibrationCallback()
quantizer = Quantizer(mod, params, [AverageMaxPerChannelConv2DBiasAddPattern(cc), AverageMaxPerChannelConv2DPattern(cc), AverageMaxPerChannelDensePattern(cc)])
random_dataset_manager = RandomDatasetManager(input_shape, 'float32', 3)

calibrater = Calibrater(quantizer, target='llvm', ctx=tvm.cpu(), dataset_manager=random_dataset_manager)
calibrated_mod = calibrater.calibrate()
print("Done calibrating")
requantized_mod = Requantizer().requantize(calibrated_mod)


with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(requantized_mod, target='llvm')

from tvm.contrib import graph_runtime
input_np = np.random.randn(*input_shape).astype('float32')

gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
gmod.set_input(**params)
gmod.set_input(input_name, input_np)
gmod.run()
out = gmod.get_output(0).asnumpy()
print("Unquantized Output:")
print(out)


print(" ___________ ")
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
q_gmod.set_input(input_name, input_np)
q_gmod.set_input(**params)
q_gmod.run()
q_out = q_gmod.get_output(0).asnumpy()
print("Quantized output:")
print(q_out)
