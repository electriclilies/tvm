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

from tvm.relay.transform.quantize import CalibrationCallback
from tvm.relay.dataflow_pattern import is_op, wildcard, DFPatternCallback, _DFPatternCallback
from tvm.relay.dataflow_pattern import ffi as pattern_ffi
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.utils import get_pad_tuple2d
from . import _ffi as ffi

import numpy as np

class QuantizerPattern(DFPatternCallback):
    # Counts the number of times we've added a scale and zp for variable naming
    scales_count = 0
    zp_count = 0

    def __init__(self, calibration_callback : CalibrationCallback = None):

        super().__init__()
        self.calibration_callback = calibration_callback

    def calibrate_pattern(self, calibration_info):
        return self.calibration_callback.calibrate_pattern(calibration_info)

    def callback(self, pre, post, node_map):
        raise NotImplementedError

    # Helper to construct the relay scale variable in a pattern
    def scale(self, name):
        var = relay.var(str(name) + "_scale_" + str(QuantizerPattern.scales_count), shape=(), dtype='float32')
        QuantizerPattern.scales_count += 1
        return var

    # Helper to construct the relay zero_point variable in a pattern
    def zero_point(self, name):
        var = relay.var(str(name) + "_zero_pt_" + str(QuantizerPattern.zp_count), shape=(), dtype='int32')
        QuantizerPattern.zp_count += 1
        return var

class Conv2DPattern(QuantizerPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.input = wildcard()
        self.conv_weight = wildcard()

        self.inputs=[self.input, self.conv_weight]

        self.conv2d = is_op('nn.conv2d')(self.input, self.conv_weight)
        
        self.pattern = self.conv2d

        self.attrs = None
        self.weight_channel_axis = None
        self.data_channel_axis = None
        self.channels = None

    def get_kernel_size(self, kernel_shape, kernel_layout):
        if kernel_layout == "OIHW":
            kernel_size = tuple(kernel_shape[2:4])
        elif kernel_layout == "HWIO":
            kernel_size = tuple(kernel_shape[0:2])
        else:
            raise ValueError("Quantizting kernel layout %s for conv2d is not yet supported. Please use OIHW or HWIO", kernel_layout)
        return kernel_size

    def get_attrs(self, attrs, kernel_shape):
        new_attr_dict = {}
        kernel_layout = attrs["kernel_layout"]
        data_layout = attrs["data_layout"]
        
        if kernel_layout == "OIHW":
            self.weight_channel_axis = 0
        elif kernel_layout == "HWIO":
            self.weight_channel_axis = 3
        else:
            raise ValueError("Quantizing kernel layout %s for conv2d is not yet supported. Please use OIHW or HWIO", kernel_layout)

        if data_layout == "NCHW":
            self.data_channel_axis = 1
        elif data_layout == "NHWC":
            self.data_channel_axis = 3
        else:
            raise ValueError("Quantizing data layout %s for conv2d is not yet supported. Please use NCHW or NHWC", data_layout)
        
        for attr in attrs.keys():
            attr_value = attrs[attr]
            if isinstance(attr_value, tvm.ir.container.Array):
                attr_value = tuple(attr_value)
            if attr == 'kernel_size':
                kernel_size = attrs[attr]
                if kernel_size is None:
                    kernel_size = self.get_kernel_size(kernel_layout, kernel_shape)
                else:
                    kernel_size = tuple([k.value for k in attrs[attr]])
                new_attr_dict[attr] = kernel_size
            elif attr == 'channels':
                self.channels = attrs[attr]
                if self.channels is None:
                    self.channels = kernel_shape[self.weight_channel_axis]
                if (isinstance(self.channels, tvm.tir.expr.IntImm)):
                    self.channels = self.channels.value
                new_attr_dict[attr] = self.channels
            elif attr == 'padding':
                self.padding = attrs[attr] # Don't need to put padding in attr dict because we explicitly construct padding
            else:
                new_attr_dict[attr] = attr_value

        new_attr_dict['out_dtype'] = 'int32'
        self.attrs = new_attr_dict

    def create_scale_zps(self):
        # Create quantization variables for arguments to this convolution.
        data_scale = self.scale('conv2d_data') 
        data_zp = self.zero_point('conv2d_data')
        weight_scale = self.scale('conv2d_weight')
        weight_zp = self.zero_point('conv2d_weight')
        self.scale_zps = [data_scale, data_zp, weight_scale, weight_zp]
    
    def quantize_args(self):
        # Quantize the data and construct args for qnn.conv2d
        quantized_data = relay.qnn.op.quantize(self.args[0], self.scale_zps[0], self.scale_zps[1], axis=self.data_channel_axis)
        quantized_weight = relay.qnn.op.quantize(self.args[1], self.scale_zps[2], self.scale_zps[3], axis=self.weight_channel_axis)
        self.quantized_args = [quantized_data, quantized_weight]
        
    def create_conv(self, args):
        return relay.qnn.op.conv2d(*args, **self.attrs)

    def callback(self, pre, post, node_map):
        self.args = [node_map[i][0] for i in self.inputs]
        conv2d = node_map[self.conv2d][0]

        # TODO: put the layout stuff in here
        self.out_dtype = conv2d.checked_type.dtype

        self.get_attrs(conv2d.attrs, infer_type(self.args[1]).checked_type.shape)

        self.create_scale_zps()
        self.quantize_args()
        # Quantize bias the same way conv is quantized
        conv_scale = self.scale_zps[0] * self.scale_zps[2] # data_scale * weight_scale
        # Conv zp is zero since QNN deals with input zps for us
        conv_zp = relay.const(0, dtype='int32')
        args = self.quantized_args[0:2] + [self.scale_zps[i] for i in [1, 3, 0, 2]] #[quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]

        if self.padding is not None:
            #TODO: Need to make this work with other layouts.
            top, left, bottom, right = [p.value for p in get_pad_tuple2d(self.padding)]
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
            pad_val = 0
            args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

        # Construct quantized qnn.conv2d and dequantize
        qnn_call = self.create_conv(args)
        dequantized_call = relay.qnn.op.dequantize(qnn_call, conv_scale, conv_zp, out_dtype=self.out_dtype, axis=self.data_channel_axis)

        return dequantized_call

class Conv2DBiasAddPattern(Conv2DPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.bias_weight = wildcard()
        self.inputs.append(self.bias_weight)
        self.pattern = is_op('nn.bias_add')(self.conv2d, self.bias_weight)
    
    def create_args(self):
        self.create_args()
        quantized_bias = relay.qnn.op.quantize(self.args[2], self.scale_zps[0], self.scale_zps[1], axis=self.data_channel_axis)
        self.quantized_args.append(quantized_bias)

    def create_conv(self, args):
        qnn_call =  relay.qnn.op.conv2d(*args, **self.attrs)
        bias_add = relay.op.nn.bias_add(qnn_call, self.quantized_args[2])
        return bias_add

class DensePattern(QuantizerPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.data = wildcard()
        self.weight = wildcard()

        self.inputs = [self.data, self.weight]
        
        self.dense = is_op('nn.dense')(self.data, self.weight)

        self.pattern = self.dense

    def get_attrs(self, attrs, weight_shape):
        self.attrs = {}
        units = attrs['units']
        if units is None:
            units = weight_shape[0]
        units = units.value
        self.attrs['units'] = units

    def create_scale_zps(self):
        # Create quantization parameters for arguments to this dense layer.
        data_scale = self.scale('dense_data') 
        data_zp = self.zero_point('dense_data')
        weight_scale = self.scale('dense_weight')
        weight_zp = self.zero_point('dense_weight')
        self.scale_zps = [data_scale, data_zp, weight_scale, weight_zp]
        
    def quantize_args(self):
        # Quantize data and construct args for qnn.dense
        quantized_data = relay.qnn.op.quantize(self.args[0], self.scale_zps[0], self.scale_zps[1])
        quantized_weight = relay.qnn.op.quantize(self.args[1], self.scale_zps[2], self.scale_zps[3], axis=0) # Axis = 0 for per channel quantization
        self.quantized_args = [quantized_data, quantized_weight]

    def create_dense(self, args):
        qnn_call = relay.qnn.op.dense(*args, **self.attrs)
        return qnn_call

    def callback(self, pre, post, node_map):
        self.args = [node_map[i][0] for i in self.inputs]
        weight = node_map[self.weight][0]

        dense = node_map[self.dense][0]
        out_dtype = dense.checked_type.dtype

        self.get_attrs(dense.attrs, infer_type(weight).checked_type.shape)

        self.create_scale_zps()
        self.quantize_args()

        args = self.quantized_args[0:2] + [self.scale_zps[i] for i in [1, 3, 0, 2]] #[quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]

        qnn_call = self.create_dense(args)
        
        dequantized_call = relay.qnn.op.dequantize(qnn_call, self.scale_zps[0] * self.scale_zps[2], relay.const(0, dtype='int32'), out_dtype=out_dtype, axis=1) # Dequantize axis = 1

        return dequantized_call

class AddPattern(QuantizerPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.lhs = wildcard()
        self.rhs = wildcard()

        self.add = is_op('add')(self.lhs, self.rhs)
        self.pattern = self.add
    
    def callback(self, pre, post, node_map):
        lhs = node_map[self.lhs][0]
        rhs = node_map[self.rhs][0]

        add = node_map[self.add][0]

        out_dtype = add.checked_type.dtype

        # Create quantization parameters for arguments to this addition
        lhs_scale = self.scale('add_lhs') 
        lhs_zp = self.zero_point('add_lhs')
        rhs_scale = self.scale('add_rhs')
        rhs_zp = self.zero_point('add_rhs')

        # Quantize, dequantize, and requantize inputs to have scale lhs_scale + rhs_scale
        # (Scale represents the lowest possible value representable in the quantized type,
        # so the smallest representable output is lhs_scale + rhs_scale)
                
        # We do this to avoid the requantize op in qnn's add, which causes issues with compilation
        # Requantize will be inserted in a future pass
        quantized_lhs = relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp)
        quantized_rhs = relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)

        dequantized_lhs = relay.qnn.op.dequantize(quantized_lhs, lhs_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)
        dequantized_rhs = relay.qnn.op.dequantize(quantized_rhs, rhs_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)

        add_scale = relay.op.add(lhs_scale, rhs_scale)

        requantized_lhs = relay.qnn.op.quantize(dequantized_lhs, add_scale, relay.const(0, dtype='int32'))
        requantized_rhs = relay.qnn.op.quantize(dequantized_rhs, add_scale, relay.const(0, dtype='int32'))
        
        add = relay.op.add(requantized_lhs, requantized_rhs)
        dequantized_call = relay.qnn.op.dequantize(add, add_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)

        return dequantized_call

class MultiplyPattern(QuantizerPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.lhs = wildcard()
        self.rhs = wildcard()

        self.multiply = is_op('multiply')(self.lhs, self.rhs)
        self.pattern = self.multiply
    
    def callback(self, pre, post, node_map):
        print("Multiply callback worked")
        lhs = node_map[self.lhs][0]
        rhs = node_map[self.rhs][0]

        multiply = node_map[self.multiply][0]
        # TODO: do I need infer_type here? 
        out_dtype = multiply.checked_type.dtype

        # Create quantization parameters for arguments to this multiplication.
        lhs_scale = self.scale('mul_lhs')
        lhs_zp = self.zero_point('mul_lhs')
        rhs_scale = self.scale('mul_rhs')
        rhs_zp = self.zero_point('mul_rhs')

        # Quantize inputs and construct args for multiply
        quantized_lhs = tvm.relay.cast(relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp), 'int32')
        quantized_rhs = tvm.relay.cast(relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp), 'int32')

        # Use normal relay multiply instead of qnn multiply to avoid requantize in qnn.mul
        # Subtract zero points to center on zero so that we can multiply lhs, rhs directly
        zeroed_quantized_lhs = relay.op.subtract(quantized_lhs, lhs_zp)
        zeroed_quantized_rhs = relay.op.subtract(quantized_rhs, rhs_zp)
                
        multiply = relay.op.multiply(zeroed_quantized_lhs, zeroed_quantized_rhs)
        dequantized_call = relay.qnn.op.dequantize(multiply, lhs_scale * rhs_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)

        return dequantized_call

class PerChannelPattern():
    def extract_attrs(self, pre, post, node_map):
        raise NotImplementedError()
    
    def attr_callback(self, expr):
        pattern_ffi.rewrite([_DFPatternCallback(self.pattern, self.extract_attrs, self.require_type)], infer_type(expr), tvm.ir.IRModule(), False)

def partition_outputs(expr):
    return ffi.partition_outputs(expr)
def rewrite_partitions(callbacks, expr):
    return ffi.rewrite_partitions([_DFPatternCallback(callback.pattern, callback.callback, callback.require_type) for callback in callbacks], infer_type(expr))
def lower_partitions(expr):
    return ffi.lower_partitions(expr)
def lower_partitions(expr):
    return ffi.lower_partitions(expr)
def skip_partitions(expr, skip_first = True, skip_last = True):
    return ffi.skip_partitions(expr, skip_first, skip_last)

