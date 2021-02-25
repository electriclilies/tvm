# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Prerequisites for this tutorial -- onnx resnet-18 and imagenet testing dataset

# Preprocess the imagenet dataset
import gluoncv
from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

import numpy as np
# source: https://github.com/onnx/models/blob/master/vision/classification/imagenet_preprocess.py
transform_fn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 2
test_data = DataLoader(ImageNet(train=False, root='~/.mxnet/datasets/imagenet').transform_first(transform_fn), batch_size=batch_size, shuffle=True)

print("Finished preprocessing imagenet... ")

# Write the dataset manager wrapper
from tvm.data import DatasetManager

class MxnetLoader(DatasetManager):
    def __init__(self, data_loader, batch_size, total_batches):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
        self.batch_sz = batch_size
        self.total_batches = total_batches
        self.idx = 0

    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1
        data, label = next(self.iter)
        return [data.asnumpy()], label

    def batch_size(self):
        return self.batch_sz

    def num_batches(self):
        return self.total_batches

    def is_empty(self):
        return self.idx >= self.total_batches

    def reset(self):
        self.idx = 0
        self.iter = iter(self.data_loader)

num_batches = 10
imagenet = MxnetLoader(test_data, batch_size, num_batches)

# Load the onnx model
import onnx
from tvm import relay

onnx_model = onnx.load(
     "/home/lorthsmith/tvm/tutorials/quantization/resnet18-v1-7.onnx")
input_dict = {"data": [batch_size, 3, 224, 224]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)

# Set up calibration
import tvm
import tvm.relay.transform.quantize as q

cc = q.AverageMaxCalibrationCallback()

quantizer = q.Quantizer(mod['main'], params, [q.AverageMaxPerChannelConv2DBiasAddPattern(cc),
         q.AverageMaxPerChannelConv2DPattern(cc),
         q.AverageMaxPerChannelDensePattern(cc),
         #q.AddPattern(cc),
         #q.MultiplyPattern(cc),
     ],
     skip_first=True,
     skip_last=True,
)

calibrator = q.QuantizationCalibrator(
     quantizer, target="llvm", ctx=tvm.cpu(), dataset_manager=imagenet, show_scale_zps=True)
print("Quantized")
calibrated_func = calibrator.calibrate()
print("Calibrating..")
requantized_func = q.Requantizer().requantize(calibrated_func)
print("Requantized")
# Build the final function
requantized_mod = tvm.IRModule.from_expr(requantized_func)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
     lib = relay.build(mod, params=params, target="llvm")
     q_lib = relay.build(requantized_mod, params=params, target="llvm")
     c_lib = relay.build(tvm.IRModule.from_expr(calibrated_func), params=params, target="llvm")

print("Built final fn")
# Take a look at accuracy

from tvm.contrib import graph_runtime

imagenet2 = MxnetLoader(test_data, batch_size, 10)

q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
c_gmod = graph_runtime.GraphModule(c_lib["default"](tvm.cpu()))
q_correct = 0
c_correct = 0
correct = 0
total = 0


while not imagenet2.is_empty():
    images, labels = imagenet2.get_next_batch()
    
    q_gmod.set_input(**{'data': images[0]})
    q_gmod.run()
    q_out = q_gmod.get_output(0).asnumpy()
    
    gmod.set_input(**{'data': images[0]})
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    c_gmod.set_input(**{'data': images[0]})
    c_gmod.run()
    c_out = gmod.get_output(0).asnumpy()

    q_predicted_labels = np.argmax(q_out, axis=1)
    c_predicted_labels = np.argmax(c_out, axis=1)
    predicted_labels = np.argmax(out, axis=1)

    print("Int8 labels: ", q_predicted_labels)
    print("No R Int8 labels: ", c_predicted_labels)
    print("Float32 labels: ", predicted_labels)
    print("Actual labels: ", labels)
    c_correct += np.sum(c_predicted_labels == labels.asnumpy())
    q_correct += np.sum(q_predicted_labels == labels.asnumpy())
    correct += np.sum(predicted_labels == labels.asnumpy())

    total += batch_size

print("calibrated int8 percent correct: ", (c_correct / total) * 100)
print("Int8 percent correct: ", (q_correct / total) * 100)
print("Float32 percent correct: ", (correct / total) * 100)
print("Difference: ", (((correct / total) * 100) - ((q_correct / total) * 100)))