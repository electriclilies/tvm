# Demo based on code from https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.new_quantize import Quantizer, Calibrater, AverageMaxCalibrationCallback, DatasetManager, Requantizer, AverageMaxPerChannelConv2DBiasAddPattern, AverageMaxPerChannelConv2DPattern, DensePattern, AddPattern, MultiplyPattern, AverageMaxPerChannelConv2DBiasAddPattern, AverageMaxPerChannelConv2DPattern, AverageMaxPerChannelDensePattern

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
# For "training", it seems like batch size 10 and n batches = 5000 works pretty well
batch_size = 2
train_dataset_manager = NumpyDatasetManager(train_images, np.ndarray.flatten(train_labels), batch_size, n_batches=1)
test_dataset_manager = NumpyDatasetManager(test_images, np.ndarray.flatten(test_labels), batch_size, n_batches=50)

# Load onnx model (model obtained from https://www.tensorflow.org/tutorials/images/cnn), exported to onnx
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/demos/cifar-model.onnx')
input_dict = {'conv2d_input:0': [batch_size, 32, 32, 3]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)

cc = AverageMaxCalibrationCallback()
#quantizer = Quantizer(mod, params, [AverageMaxPerChannelConv2DBiasAddPattern(cc), AverageMaxPerChannelConv2DPattern(cc), DensePattern(cc), AddPattern(cc), MultiplyPattern(cc)]) #[Conv2DBiasAddPattern(cc), Conv2DPattern(cc), DensePattern(cc), AddPattern(cc), MultiplyPattern(cc)])
quantizer = Quantizer(mod, params, [AverageMaxPerChannelConv2DPattern(cc), AverageMaxPerChannelDensePattern(cc), AddPattern(cc)], skip_last=False)#, AddPattern(cc), MultiplyPattern(cc)], skip_last=False)
calibrater = Calibrater(quantizer, target='llvm', ctx=tvm.cpu(), dataset_manager=train_dataset_manager)
calibrated_mod = calibrater.calibrate()
print("Calibrated mod: ")
print(relay.transform.InferType()(calibrated_mod).astext(False))

print("Requantizing...")
requantized_mod = Requantizer().requantize(calibrated_mod)
print("Requantized mod: \n", requantized_mod.astext(False))

print("Calculating accuracy...")
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(requantized_mod, target='llvm')


from tvm.contrib import graph_runtime
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
q_correct = 0
correct = 0
total = 0

while not test_dataset_manager.is_empty():
    image_list, label = test_dataset_manager.get_next_batch()
    q_gmod.set_input(**{'conv2d_input:0': image_list[0]})
    q_gmod.run()
    q_out = q_gmod.get_output(0).asnumpy()

    gmod.set_input(**{'conv2d_input:0': image_list[0]})
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    q_predicted_labels = np.argmax(q_out, axis=1)
    predicted_labels = np.argmax(out, axis=1)

    print("Int8 labels: ", q_predicted_labels)
    print("Float32 labels: ", predicted_labels)
    print("Actual labels: ", label)
    print()

    q_correct += np.sum(q_predicted_labels == label)
    correct += np.sum(predicted_labels == label)
    total += batch_size

print("Int8 percent correct: ", (q_correct / total) * 100)
print("Float32 percent correct: ", (correct / total) * 100)
print("Difference: ", (((correct / total) * 100) - ((q_correct / total) * 100)))
