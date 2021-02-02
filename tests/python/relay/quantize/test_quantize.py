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
from tvm.relay.transform.quantize import Quantizer, Conv2DPattern, Conv2DBiasAddPattern, DensePattern, AddPattern, MultiplyPattern
from tvm.relay.op.nn.utils import get_pad_tuple2d
from tvm.relay.frontend.common import infer_type
import numpy as np

def quantize_and_check(pre_func, expected_func, quantizer_pattern_list, skip_first=False, skip_last=False):
    infer_type(pre_func)
    quantizer = Quantizer(pre_func, None, quantizer_pattern_list, skip_first=skip_first, skip_last=skip_last)
    infer_type(quantizer.quantized_func)
    assert tvm.ir.structural_equal(quantizer.quantized_func, expected_func)

def create_scale_zps(channels=None):
    data_scale_var = relay.var("data_scale", shape=(), dtype='float32')
    data_zp_var = relay.var("data_zp", shape=(), dtype='int32')

    if not channels:
        weight_scale_var = relay.var("weight_scale", shape=(), dtype='float32')
        weight_zp_var = relay.var("weight_zp", shape=(), dtype='int32')
    else:
        weight_scale_var = relay.var("weight_scale", shape=(channels,), dtype='float32')
        weight_zp_var = relay.var("weight_zp", shape=(channels,), dtype='int32')

    return data_scale_var, data_zp_var, weight_scale_var, weight_zp_var

def test_conv2d(data_shape, weight_shape, attrs):
    
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))

    # Pre quantize input
    conv2d = relay.op.nn.conv2d(data, weight, **attrs)
    pre_func = relay.Function([data, weight], conv2d)

    kernel_layout = attrs["kernel_layout"]
    data_layout = attrs["data_layout"]

    if kernel_layout == "OIHW":
        kernel_size = tuple(weight_shape[2:4])
        weight_channel_axis = 0
    elif kernel_layout == "HWIO":
        kernel_size = tuple(weight_shape[0:2])
        weight_channel_axis = 3
    else:
        raise ValueError("We don't support layouts other than OIHW or HWIO, but got %s. Please provide a compatible layout to the test. ", kernel_layout)

    if data_layout == "NCHW":
        data_channel_axis = 1
    elif data_layout == "NHWC":
        data_channel_axis = 3
    else:
        raise ValueError("We don't support layouts other than NCHW or NHWC, but got %s. Please provide a compatible layout to the test. ", data_layout)
        
    # Post quantize output
    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps()

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var, axis=data_channel_axis) # Put axis in
    q_weight = relay.qnn.op.quantize(weight, weight_scale_var, weight_zp_var, axis=weight_channel_axis)
    
    if 'padding' in attrs.keys():
        padding = attrs['padding']
    else:
        padding = None

    if padding is not None:
        top, left, bottom, right = get_pad_tuple2d(padding)
        if kernel_layout == 'OIHW':
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
        elif kernel_layout == 'HWIO':
            pad_width = ((top, bottom), (left, right), (0, 0), (0, 0),)
        pad_val = 0
        q_data = relay.op.nn.pad(q_data, pad_width, pad_val)
    
    q_conv2d = relay.qnn.op.conv2d(q_data, q_weight, data_zp_var, weight_zp_var, data_scale_var, weight_scale_var, data_layout=data_layout, kernel_layout=kernel_layout, kernel_size=kernel_size, channels=weight_shape[weight_channel_axis])

    deq_conv2d = relay.qnn.op.dequantize(q_conv2d, data_scale_var * weight_scale_var, relay.const(0, dtype='int32'), out_dtype="float32", axis=data_channel_axis)
    quantized_func = relay.Function([data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_conv2d)
    quantize_and_check(pre_func, quantized_func, [Conv2DPattern(None)])

def test_conv2d_bias(data_shape, weight_shape, bias_shape, attrs): 
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    bias = relay.const(np.random.rand(*bias_shape).astype('float32'), "float32")

    kernel_layout = attrs["kernel_layout"]
    data_layout = attrs["data_layout"]
    if kernel_layout == "OIHW":
        kernel_size = tuple(weight_shape[2:4])
        weight_channel_axis = 0
    elif kernel_layout == "HWIO":
        kernel_size = tuple(weight_shape[0:2])
        weight_channel_axis = 3
    else:
        raise ValueError("We don't support layouts other than OIHW or HWIO, but got %s. Please provide a compatible layout to the test. ", kernel_layout)

    if data_layout == "NCHW":
        data_channel_axis = 1
    elif data_layout == "NHWC":
        data_channel_axis = 3
    else:
        raise ValueError("We don't support layouts other than NCHW or NHWC, but got %s. Please provide a compatible layout to the test. ", data_layout)

    # Pre quantize input
    conv2d = relay.op.nn.conv2d(data, weight, **attrs)
    #bias_add = relay.op.nn.bias_add(conv2d, bias, axis=data_channel_axis)
    bias_add = relay.op.nn.bias_add(conv2d, bias, axis=data_channel_axis)

    pre_func = relay.Function([data, weight], bias_add)

    # Post quantize output
    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps()

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var, axis=data_channel_axis) # Put axis in
    q_weight = relay.qnn.op.quantize(weight, weight_scale_var, weight_zp_var, axis=weight_channel_axis)

    if 'padding' in attrs.keys():
        padding = attrs['padding']
    else:
        padding = None

    if padding is not None:
        top, left, bottom, right = get_pad_tuple2d(padding)
        if kernel_layout == 'OIHW':
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
        elif kernel_layout == 'HWIO':
            pad_width = ((top, bottom), (left, right), (0, 0), (0, 0),)
        pad_val = 0
        q_data = relay.op.nn.pad(q_data, pad_width, pad_val)
    
    q_conv2d = relay.qnn.op.conv2d(q_data, q_weight, data_zp_var, weight_zp_var, data_scale_var, weight_scale_var, data_layout=data_layout, kernel_layout=kernel_layout, kernel_size=kernel_size, channels=weight_shape[weight_channel_axis])
    bias_add = relay.op.nn.bias_add(q_conv2d, relay.qnn.op.quantize(bias, data_scale_var, data_zp_var, axis=0, out_dtype='int32'), axis=data_channel_axis)

    deq_conv2d = relay.qnn.op.dequantize(bias_add, data_scale_var * weight_scale_var, relay.const(0, dtype='int32'), out_dtype="float32", axis=data_channel_axis)
    quantized_func = relay.Function([data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_conv2d)
    quantize_and_check(pre_func, quantized_func, [Conv2DBiasAddPattern(None)])


def test_dense(data_shape, weight_shape, attrs):
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    pre_func = relay.Function([data, weight], relay.nn.dense(data, weight))

    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps()

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var)
    q_weight = relay.qnn.op.quantize(weight, weight_scale_var, weight_zp_var, axis=0)

    q_dense = relay.qnn.op.dense(q_data, q_weight, data_zp_var, weight_zp_var, data_scale_var, weight_scale_var, **attrs)
    deq_dense = relay.qnn.op.dequantize(q_dense, data_scale_var * weight_scale_var, relay.const(0, dtype='int32'), axis=1)

    quantized_func = relay.Function([data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_dense)
    quantize_and_check(pre_func, quantized_func, [DensePattern(None)])

def test_add(lhs_shape, rhs_shape):
    lhs = relay.var("lhs", relay.TensorType(lhs_shape, dtype='float32'))
    rhs = relay.var("rhs", relay.TensorType(rhs_shape, dtype='float32'))
    pre_func = relay.Function([lhs, rhs], relay.add(lhs, rhs))

    lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var = create_scale_zps()
    q_lhs = relay.qnn.op.quantize(lhs, lhs_scale_var, lhs_zp_var)
    q_rhs = relay.qnn.op.quantize(rhs, rhs_scale_var, rhs_zp_var)

    deq_lhs = relay.qnn.op.dequantize(q_lhs, lhs_scale_var, relay.const(0, dtype='int32'))
    deq_rhs = relay.qnn.op.dequantize(q_rhs, rhs_scale_var, relay.const(0, dtype='int32'))

    add_scale = relay.op.add(lhs_scale_var, rhs_scale_var)

    requantized_lhs = relay.qnn.op.quantize(deq_lhs, add_scale, relay.const(0, dtype='int32'))
    requantized_rhs = relay.qnn.op.quantize(deq_rhs, add_scale, relay.const(0, dtype='int32'))

    add = relay.op.add(requantized_lhs, requantized_rhs)
    deq_add = relay.qnn.op.dequantize(add, add_scale, relay.const(0, dtype='int32'))

    quantized_func = relay.Function([lhs, rhs, lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var], deq_add)
    quantize_and_check(pre_func, quantized_func, [AddPattern(None)])

def test_mul(lhs_shape, rhs_shape):
    lhs = relay.var("lhs", relay.TensorType(lhs_shape, dtype='float32'))
    rhs = relay.var("rhs", relay.TensorType(rhs_shape, dtype='float32'))
    pre_func = relay.Function([lhs, rhs], relay.multiply(lhs, rhs))

    lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var = create_scale_zps()
    q_lhs = relay.qnn.op.quantize(lhs, lhs_scale_var, lhs_zp_var)
    q_rhs = relay.qnn.op.quantize(rhs, rhs_scale_var, rhs_zp_var)

    zeroed_q_lhs = relay.op.subtract(relay.op.cast(q_lhs, "int32"), lhs_zp_var)
    zeroed_q_rhs = relay.op.subtract(relay.op.cast(q_rhs, "int32"), rhs_zp_var)

    multiply = relay.op.multiply(zeroed_q_lhs, zeroed_q_rhs)
    deq_multiply = relay.qnn.op.dequantize(multiply, lhs_scale_var * rhs_scale_var, relay.const(0, dtype='int32'))

    quantized_func = relay.Function([lhs, rhs, lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var], deq_multiply)
    quantize_and_check(pre_func, quantized_func, [MultiplyPattern(None)])

def test_skip_layers(data_shape, weight_shape, attrs):
    # We'll test skip_layers with the dense op
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    pre_func = relay.Function([data, weight], relay.nn.dense(data, weight))

    quantize_and_check(pre_func, pre_func, [DensePattern(None)], skip_first=True, skip_last=False)
    quantize_and_check(pre_func, pre_func, [DensePattern(None)], skip_first=False, skip_last=True)
    quantize_and_check(pre_func, pre_func, [DensePattern(None)], skip_first=True, skip_last=True)

if __name__ == "__main__":
    test_conv2d((2, 3, 32, 32), (32, 3, 3, 3), {'kernel_size': [3, 3], 'kernel_layout': 'OIHW', 'data_layout': 'NCHW', 'padding': [0, 0, 0, 0]})
    test_conv2d((2, 32, 32, 3), (3, 3, 3, 32), {'kernel_size': [3, 3], 'kernel_layout': 'HWIO', 'data_layout': 'NHWC', 'padding': [0, 0, 0, 0]})

    test_conv2d_bias((2, 3, 32, 32), (32, 3, 3, 3), (32,), {'kernel_size': [3, 3], 'kernel_layout': 'OIHW', 'data_layout': 'NCHW', 'padding': [0, 0, 0, 0]})
    test_conv2d_bias((2, 32, 32, 3), (3, 3, 3, 32), (32,), {'kernel_size': [3, 3], 'kernel_layout': 'HWIO', 'data_layout': 'NHWC', 'padding': [0, 0, 0, 0]})

    test_dense((1, 8), (16, 8), {'units': 16})
    test_dense((1, 4), (3, 4), {'units': 3})

    test_add((1, 2, 3), (1, 2, 3))

    test_mul((1, 2, 3), (1, 2, 3))

    test_skip_layers((1, 8), (16, 8), {'units': 16})
    test_skip_layers((1, 4), (3, 4), {'units': 3})

