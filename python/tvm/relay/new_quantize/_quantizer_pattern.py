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

from tvm.relay.new_quantize import DefaultCalibrater
from tvm.relay.dataflow_pattern import is_op, wildcard, DFPatternCallback, _DFPatternCallback
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.utils import get_pad_tuple2d
from . import _ffi as ffi

import numpy as np

class QuantizerPattern(DFPatternCallback):
    # Counts the number of times we've added a scale and zp for variable naming
    scales_count = 0
    zp_count = 0

    def __init__(self, calibrator : DefaultCalibrater = None):
        super().__init__()
        self.calibrator = calibrator

    def calibrate_pattern(self, *args):
        return self.calibrator.calibrate_pattern(*args)

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

# TODO: where should these live?

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

class Conv2DPattern(QuantizerPattern):
    def __init__(self, calibrator : DefaultCalibrater = None):
        super().__init__(calibrator)
        self.input = wildcard()
        self.conv_weight = wildcard()

        self.conv2d = is_op('nn.conv2d')(self.input, self.conv_weight)
        
        self.pattern = self.conv2d

    def callback(self, pre, post, node_map):
        data = node_map[self.input][0]
        weight = node_map[self.conv_weight][0]
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
        args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]

        if padding is not None:
            #TODO: Need to make this work with other layouts.
            top, left, bottom, right = [p.value for p in get_pad_tuple2d(padding)]
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
            pad_val = 0
            args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

        # Construct quantized qnn.conv2d and dequantize
        qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
        dequantized_call = relay.qnn.op.dequantize(qnn_call, conv_scale, conv_zp, out_dtype=out_dtype, axis=data_channel_axis)

        return dequantized_call

class DensePattern(QuantizerPattern):
    def __init__(self, calibrater : DefaultCalibrater):
        super().__init__(calibrater)
        self.data = wildcard()
        self.weight = wildcard()
        
        self.dense = is_op('nn.dense')(self.data, self.weight)

        self.pattern = self.dense

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        weight = node_map[self.weight][0]

        dense = node_map[self.dense][0]
        
        attrs = dense.attrs
        out_dtype = dense.checked_type.dtype

        # Create quantization parameters for arguments to this dense layer.
        data_scale = self.scale('dense_data') 
        data_zp = self.zero_point('dense_data')
        weight_scale = self.scale('dense_weight')
        weight_zp = self.zero_point('dense_weight')
                
        # Quantize data and construct args for qnn.dense
        quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
        quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp, axis=0) # Axis = 0 for per channel quantization

        args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]

        new_attr_dict = {}
        units = attrs['units']
        if units is None:
            weight_type_info = infer_type(weight)
            units = weight_type_info.checked_type.shape[0]
        units = units.value
        new_attr_dict['units'] = units

        qnn_call = relay.qnn.op.dense(*args, **new_attr_dict)
        dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype, axis=1) # Dequantize axis = 1

        return dequantized_call

# Maybe is that multiply doesnt match any patterns
# add also doesnt match but it doesnt break everything
class AddPattern(QuantizerPattern):
    def __init__(self, calibrater : DefaultCalibrater):
        super().__init__(calibrater)
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
    def __init__(self, calibrater : DefaultCalibrater):
        super().__init__(calibrater)
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

def partition_outputs(expr):
    return ffi.partition_outputs(expr)
def rewrite_partitions(callbacks, expr):
    return ffi.rewrite_partitions([_DFPatternCallback(callback.pattern, callback.callback, callback.require_type) for callback in callbacks], infer_type(expr))
def lower_partitions(expr):
    return ffi.lower_partitions(expr)