
import tvm
from tvm import relay
import torch

from torchvision.models import resnet
from tvm.relay.new_quantize import Quantizer, DatasetManager, AverageMeanCalibrater

import numpy as np
# TODO: importing resnet from torchvision.models causes protobuf error if before importing stuff from new_quantize????

class RandomDatasetManager(DatasetManager):
    def __init__(self, data_shape, dtype, num_batches):
        self.idx = 0
        self.data_shape = data_shape
        self.dtype = dtype
        self.num_batches = num_batches
    
    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1
        return np.random.randn(*self.data_shape).astype(self.dtype), None

    def num_batches(self):
        return self.num_batches

    def is_empty(self):
        return self.idx >= self.num_batches

    def reset(self):
        self.idx = 0

pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shape = (1, 3, 224, 224)
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)
quantized_mod, calibration_map = Quantizer().quantize(mod, params)

random_dataset_manager = RandomDatasetManager(input_shape, 'float32', 2000)
average_mean_calibrater = AverageMeanCalibrater(random_dataset_manager)
calibrated_mod = average_mean_calibrater.calibrate(quantized_mod, calibration_map)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(calibrated_mod, target='llvm')

from tvm.contrib import graph_runtime
input_np = np.random.randn(1, 3, 224, 224).astype('float32')

gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
gmod.set_input(**params)
gmod.set_input('input', input_np)
gmod.run()
out = gmod.get_output(0).asnumpy()
print("Unquantized Output:")
print(out)


print(" ___________ ")
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
q_gmod.set_input('input', input_np)
q_gmod.set_input(**params)
q_gmod.run()
q_out = q_gmod.get_output(0).asnumpy()
print("Quantized output:")
print(q_out)
