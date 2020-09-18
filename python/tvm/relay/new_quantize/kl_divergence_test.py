import tvm
from tvm import relay
from tvm.relay.new_quantize import quantize_pass, Calibrater

import torch
from torchvision.models import resnet

import numpy as np
import math
import scipy.stats

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

    def calibration_callback(self, var_pairs, input_subgraph_fn_pairs, output_subgraph_fn_pair):
        value_dict = {}
        for ((scale_var, zp_var), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(var_pairs, input_subgraph_fn_pairs):
            
            data = self.evaluate(data_subgraph_fn, {'input', self.input}, 'llvm', tvm.cpu())
            
            # Put output at this layer into histogram, arbitrarily pick 2048 bins
            hist, bin_edges = np.histogram(data, bins=2048)
            
            divergence = []
            for i in range(128, 2048):
                
                # Reference distribution P is the original distribution
                reference_dist_p = hist[0:i-1]
                
                # Number of points outside of reference dist P
                outliers_count = np.sum(hist[i:2047])

                reference_dist_p[i-1] += outliers_count

                # Create quantized distribution by redistributing reference_dist_p
                # EX: if we have reference_dist_p = [1, 0, 2, 3, 5, 3, 1, 7], and we want to quantize to 2 bins,
                #     we merge them to get [1 + 0 + 2 + 3, 3 + 5 + 1, 7 = [6, 16]
                #     Then we proportionally expand back to 8, preserving zeros, getting
                #     Q = [6/3, 0, 6/3, 6/3, 16/4, 16/4, 16/4, 16/4]
                #     Finally we normalize and calculate KL-divergence

                # Source: https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

                # Split reference_dist_p into 128 chunks
                reshaped_p = np.reshape(reference_dist_p, ((i / 128), 128)) # TODO: How to deal w/ i not divisible by 128?
                
                # Merge reference_dist_p by summing
                Q = np.sum(reshaped_p, axis=0)

                # Count the non-zero elements in each chunk
                num_nonzero = np.count_nonzero(reshaped_p, axis=0)

                expanded_q = []

                for j in range(i):
                    if reference_dist_p[j] == 0: # Preserve 0s
                        expanded_q.append(0)
                    else:
                        expanded_q.append(num_nonzero[i/128] / Q[i/128])

                divergence.append(scipy.stats.entropy(reference_dist_p, expanded_q))

            # TODO: go thru divergence, find best i and calculate threshold, scale

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

print(calibrated_mod)

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

print("Quantized Mod: ")
print(q_gmod)