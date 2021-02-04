# Demo based on code from https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.transform.quantize import Quantizer, Calibrator, GlobalCalibrator, DatasetManager
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import onnx
import numpy as np

class NumpyDatasetManager(DatasetManager):
    # Assumes numpy_data is in form [num_inputs, c, h, w] and labels is [num_inputs]
    def __init__(self, numpy_data, numpy_labels, batch_size=1, n_batches=None):
        self.idx = 0
        self.numpy_data = numpy_data
        self.numpy_labels = numpy_labels
        assert self.numpy_data.shape[0] == self.numpy_labels.shape[0], "First dimension of data and label arrays must match."
        assert self.numpy_data.shape[0] >= batch_size, "Batch size too large. You must provide enough data points for at least one batch."
        self.batch_size = batch_size
        if n_batches is None:
            self.n_batches = numpy_data.shape[0] // self.batch_size
        else:
            assert n_batches * batch_size <= numpy_data.shape[0]
            self.n_batches = n_batches

    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        batched_data = self.numpy_data[self.idx * self.batch_size : (self.idx + 1) * self.batch_size]
        batched_label = self.numpy_labels[self.idx * self.batch_size : (self.idx + 1) * self.batch_size]
        self.idx += 1
        return [batched_data], batched_label

    def num_batches(self):
        return self.n_batches

    def is_empty(self):
        return self.idx >= self.n_batches

    def reset(self):
        self.idx = 0

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create dataset manager
batch_size = 20
train_dataset_manager = NumpyDatasetManager(train_images, np.ndarray.flatten(train_labels), batch_size, n_batches=500)
test_dataset_manager = NumpyDatasetManager(test_images, np.ndarray.flatten(test_labels), batch_size, n_batches=50)

# Load onnx model (model obtained from https://www.tensorflow.org/tutorials/images/cnn), exported to onnx
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/demos/cifar-model.onnx')
input_dict = {'conv2d_input:0': [batch_size, 32, 32, 3]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)

# Quantize

from tvm.relay.transform.quantize import Conv2DBiasAddPattern, Conv2DPattern, DensePattern, AddPattern, MultiplyPattern, partition_outputs, rewrite_partitions, lower_partitions


c = GlobalCalibrator(2.0, 0)
quantizer = Quantizer(mod['main'], [Conv2DBiasAddPattern(c), Conv2DPattern(c), DensePattern(c), AddPattern(c), MultiplyPattern(c)])
print("Quantizer created")
calibrator = Calibrator(quantizer, target='llvm', ctx=tvm.cpu())
print("Calibrator created")
calibrator.calibrate()
print("Everything worked...")
print(calibrator.calibration_info.scale_zp_value_map) 
print("Done")
exit()


callback = Conv2DBiasAddPattern()
print("-----Prior-----")
print(mod["main"])
f = callback.pattern.partition(mod["main"])
print("-----Partitioned------")
print(f)
print("-----Partitioned with Outputs------")
f = partition_outputs(f)
print("-----RewritePartitions------")
f = rewrite_partitions([callback], f)
print(f['new_out'])
#print(f['infos_'])
infos = f['infos_']
for i in infos:
    for count in range(len(i.input_scale_zps)):
        print(i.input_scale_zps[count])
        print(i.input_scale_zps[count][0])
        print(i.input_scale_zps[count][1])
