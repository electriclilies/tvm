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

import tvm
from tvm import relay

from tvm.relay.dataflow_pattern import is_op, wildcard, DFPatternCallback, _DFPatternCallback
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.utils import get_pad_tuple2d
from . import _ffi as ffi

# Target patterns
# conv2d -> biasadd
# dense -> biasadd
# conv2d
# dense
# add
# multiply
# Default calibration method
class DefaultCalibrater():

    def calibrate_pattern(self, calibration_info):
        raise NotImplementedError

# TODO: where should I live?
class GlobalCalibrater(DefaultCalibrater):
    def __init__(self, scale_value, zp_value):
        self.scale_value = scale_value
        self.zp_value = zp_value
    def calibrate_pattern(self, calibration_info):
        raise NotImplementedError

class QuantizerPattern(DFPatternCallback):
    # Counts the number of times we've added a scale and zp for variable naming
    scales_count = 0
    zp_count = 0

    def __init__(self, calibrator : DefaultCalibrater = None):
        super().__init__()
        self.calibrator = calibrator

    def calibrate_pattern(self, *args):
        self.calibrator.calibrate_pattern(*args)

    def callback(self, pre, post, node_map):
        raise NotImplementedError

    # Helper to construct the relay scale variable in a pattern
    def scale(self, name):
        #var = relay.var(str(name) + "_scale_" + str(self.scales_count), shape=(te.size_var("channels"),), dtype='float32')
        var = relay.var(str(name) + "_scale_" + str(QuantizerPattern.scales_count), shape=(), dtype='float32')
        QuantizerPattern.scales_count += 1
        return var

    # Helper to construct the relay zero_point variable in a pattern
    def zero_point(self, name):
        #var = relay.var(str(name) + "_zero_pt_" + str(self.zp_count), shape=(te.size_var("channels"),), dtype='int32')
        var = relay.var(str(name) + "_zero_pt_" + str(QuantizerPattern.zp_count), shape=(), dtype='int32')
        QuantizerPattern.zp_count += 1
        return var

#TODO : Move into a different file
class Conv2DBiasAddPattern(QuantizerPattern):
    def __init__(self, calibrator : DefaultCalibrater = None):
        super().__init__(calibrator)
        self.input = wildcard()
        self.conv_weight = wildcard()
        self.bias_weight = wildcard() 

        self.conv2d = is_op('nn.conv2d')(self.input, self.conv_weight)
        
        self.pattern = is_op('nn.bias_add')(self.conv2d, self.bias_weight)

    def callback(self, pre, post, node_map):
        data = node_map[self.input][0]
        weight = node_map[self.conv_weight][0]
        bias = node_map[self.bias_weight][0]
        conv2d = node_map[self.conv2d][0]

        # TODO: put the layout stuff in here
        attrs = conv2d.attrs
        out_dtype = conv2d.checked_type.dtype

        new_attr_dict = {}
        kernel_type_info = infer_type(weight)
        kernel_layout = attrs["kernel_layout"]
        data_layout = attrs["data_layout"]
        
        if kernel_layout == "OIHW":
            weight_channel_axis = 0
        elif kernel_layout == "HWIO":
            weight_channel_axis = 3
        else:
            raise ValueError("Quantizing kernel layout %s for conv2d is not yet supported. Please use OIHW or HWIO", kernel_layout)

        if data_layout == "NCHW":
            data_channel_axis = 1
        elif data_layout == "NHWC":
            data_channel_axis = 3
        else:
            raise ValueError("Quantizing data layout %s for conv2d is not yet supported. Please use NCHW or NHWC", data_layout)

        for attr in attrs.keys():
            attr_value = attrs[attr]
            if isinstance(attr_value, tvm.ir.container.Array):
                attr_value = tuple(attr_value)
            if attr == 'kernel_size':
                kernel_size = attrs[attr]
                if kernel_size is None:
                    if kernel_layout == "OIHW":
                        kernel_size = tuple(kernel_type_info.checked_type.shape[2:4])
                    elif kernel_layout == "HWIO":
                        kernel_size = tuple(kernel_type_info.checked_type.shape[0:2])
                    else:
                        raise ValueError("Quantizting kernel layout %s for conv2d is not yet supported. Please use OIHW or HWIO", kernel_layout)
                else:
                    kernel_size = tuple([k.value for k in attrs[attr]])
                new_attr_dict[attr] = kernel_size
            elif attr == 'channels':
                channels = attrs[attr]
                if channels is None:
                    channels = kernel_type_info.checked_type.shape[weight_channel_axis].value
                new_attr_dict[attr] = channels
            elif attr == 'padding':
                padding = attrs[attr] # Don't need to put padding in attr dict because we explicitly construct padding
            else:
                new_attr_dict[attr] = attr_value

        new_attr_dict['out_dtype'] = 'int32'

        # Create quantization variables for arguments to this convolution.
        data_scale = self.scale('conv2d_data') 
        data_zp = self.zero_point('conv2d_data')
        weight_scale = self.scale('conv2d_weight')
        weight_zp = self.zero_point('conv2d_weight')

        # Quantize the data and construct args for qnn.conv2d

        quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
        quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp, axis=weight_channel_axis)
        
        # Quantize bias the same way conv is quantized
        conv_scale = data_scale * weight_scale
        # Conv zp is zero since QNN deals with input zps for us
        conv_zp = relay.const(0, dtype='int32')
        quantized_bias = relay.qnn.op.quantize(bias, conv_scale, conv_zp)
        args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]


        if padding is not None:
            #TODO: Need to make this work with other layouts.
            top, left, bottom, right = [p.value for p in get_pad_tuple2d(padding)]
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
            pad_val = 0
            args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

        # Construct quantized qnn.conv2d and dequantize
        qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
        bias_add = relay.op.nn.bias_add(qnn_call, quantized_bias)
        dequantized_call = relay.qnn.op.dequantize(bias_add, conv_scale, conv_zp, out_dtype=out_dtype, axis=data_channel_axis)

        return dequantized_call

# Step 1: Partition the Patterns
#       Departition skipped layers
# Step 2: Analyze paritioned graph for layer counts
# Step 3: Rewrite paritioned functions
# Step 4: lower partitioned functions back into graph

def partition_outputs(expr):
    return ffi.partition_outputs(expr)
def rewrite_partitions(callbacks, expr):
    return ffi.rewrite_partitions([_DFPatternCallback(callback.pattern, callback.callback, callback.require_type) for callback in callbacks], infer_type(expr))
def lower_partitions(expr):
    return ffi.lower_partitions(expr)