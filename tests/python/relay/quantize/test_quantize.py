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
from tvm.relay.transform.quantize import Quantizer, Conv2DPattern

def quantize_and_check(before_func, after_func, quantizer_pattern_list):
    quantizer = Quantizer(before_func, None, quantizer_pattern_list) # Pass in None for params
    
    quantized_mod = quantizer.q_tuple_subgraph_mod
    assert tvm.ir.structural_equal(quantized_mod, after_func)    

def test_conv2d(data_shape, weight_shape, attrs):
    
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))

    # Pre quantize input
    conv2d = relay.op.nn.conv2d(data, weight, **attrs)
    pre_func = relay.Function(conv2d, [data, weight])

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
    data_scale_var = relay.var("data_scale", shape=(), dtype='float32')
    data_zp_var = relay.var("data_zp", shape=(), dtype='int32')

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var, axis=data_channel_axis) # Put axis in

    weight_scale_var = relay.var("weight_scale", shape=(), dtype='float32')
    weight_zp_var = relay.var("weight_zp", shape=(), dtype='int32')
    
    q_weight = relay.qnn.op.quantize(weight, weight_scale_var, weight_zp_var, axis=weight_channel_axis)

    q_conv2d = relay.qnn.op.conv2d(q_data, q_weight, data_scale_var, weight_scale_var, data_zp_var, weight_zp_var, kernel_size=kernel_size, channels=weight_shape[weight_channel_axis])

    deq_conv2d = relay.qnn.op.dequantize(q_conv2d, weight_scale_var * data_scale_var, relay.const(0, dtype='int32'), out_dtype="float32", axis=data_channel_axis)
    quantized_func = relay.Function(deq_conv2d, [data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var])

    quantize_and_check(pre_func, quantized_func, [Conv2DPattern(None)])
# Tensor[(2, 32, 32, 3), float32]) -> Tensor[(2, 10), float32] 
def test_conv2d_bias():
    pass

def test_dense():
    pass

def test_add():
    pass

def test_mul():
    pass

def test_all():
    pass

test_conv2d((4, 10, 10, 3), ())