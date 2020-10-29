import tvm
from tvm import relay
from tvm.relay.new_quantize import quantize_pass, Calibrater

import torch
from torchvision.models import resnet

import numpy as np
import math

# Calculates the KL-divergence, given two distributions p_dist (unquantized values), q_dist (quantized values)
# Currently we assume the tensor is passed in in NCHW format.
# TODO: if the p_dist and q_dist are small, it seems that normalization can introduce error. how to deal with?
def calculate_kl_divergence(p_dist, q_dist, axis=0, elem_wise=False):
    assert isinstance(p_dist, np.ndarray)
    assert isinstance(q_dist, np.ndarray)
    assert p_dist.shape == q_dist.shape

    # Normalize element-wise
    
    # TODO: for small values, normalization could introduce errors...
    normal_p_dist = np.divide(p_dist, np.sum(p_dist, axis=axis)) # KL-divergence only works on normalized distributions
    normal_q_dist = np.divide(q_dist, np.sum(q_dist, axis=axis))

    elemwise_divergence = np.sum(normal_p_dist * np.log(normal_p_dist / normal_q_dist), axis=axis) # TODO: what to do if qdist has zero

    if elem_wise:
        # Make numpy array 4d-- if given (n, c, h, w) input, elemwise_divergence.shape is (c, h, w)
        # we want (1, c, h, w)
        return np.array([elemwise_divergence])
    else:
        return np.average(elemwise_divergence)

# This really just emulates a list-- either add functionality or remove
class Dataset():
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        else: 
            data = self.dataset[self.idx]
            self.idx += 1
        return data

class KLDivergenceCalibrater(Calibrater):
    def __init__(self, input):
        super().__init__()
        self.input = input

    def calibration_callback(self, var_pairs, input_subgraph_fn_pairs, output_subgraph_fn_pair):
        value_dict = {}
        for ((scale_var, zp_var), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(var_pairs, input_subgraph_fn_pairs):
            
            quantized_data_subgraph_fn = self.bind_set_variables(quantized_data_subgraph_fn)
            
            # Set zero point to zero, since no benefit from non-zero zero points
            quantized_data_subgraph_fn = self.bind_variable(quantized_data_subgraph_fn, zp_var.name_hint, 0)

            # Calculate the KL-divergence between the unquantized data and the quantized data,
            # and record the scale with the lowest KL-divergence
            unquantized_output = self.evaluate_subgraph(data_subgraph_fn, {'input': self.input}, 'llvm', tvm.cpu())

            best_kl = math.inf # KL-divergence is between 0 and infinity (only infinity if p(x) = n but q(x) = 0)
            best_scale = 1 # TODO: change default scale... 

            for scale_value in [0.1, 0.2, 0.3, 0.4]: #TODO: pick these more intelligently
                scaled_quantized_data_subgraph_fn = self.bind_variable(quantized_data_subgraph_fn, scale_var.name_hint, scale_value)
                scaled_output = self.evaluate_subgraph(scaled_quantized_data_subgraph_fn, {'input': self.input}, 'llvm', tvm.cpu()) # TODO: change the way input var name is handled
                
                kl_divergence = calculate_kl_divergence(unquantized_output, scaled_output)
                if kl_divergence <= best_kl:
                    best_kl = kl_divergence
                    best_scale = scale_value

            value_dict[scale_var.name_hint] = np.array(best_scale).astype('float32')
            value_dict[zp_var.name_hint] = np.array(0).astype('int32')

        return value_dict

print("Test KL-divergence calibration pass: ")

# Get the model
pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
n = 2
input_shape = (n, 3, 224, 224)
n, c, h, w = input_shape
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)
quantized_mod, calibration_map = quantize_pass.quantize(mod, params, skip_layers=[])

# Construct a random dataset
np.random.seed(0)

n_input = np.random.uniform(1, 5, size=(n, c, h, w)).astype('float32')

kl = calculate_kl_divergence(n_input, np.around(n_input, decimals=2))

# Create KLDivergenceCalibrater
kl_calibrater = KLDivergenceCalibrater(n_input)
calibrated_mod = kl_calibrater.calibrate(quantized_mod, calibration_map, params)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(calibrated_mod, target='llvm')

from tvm.contrib import graph_runtime

gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
gmod.set_input(**params)
gmod.set_input('input', input)
gmod.run()
out = gmod.get_output(0).asnumpy()
print("Unquantized Output:")
print(out)

print(" ___________ ")
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
q_gmod.set_input('input', input)
q_gmod.set_input(**params)
q_gmod.run()
q_out = q_gmod.get_output(0).asnumpy()
print("Quantized output:")
print(q_out)