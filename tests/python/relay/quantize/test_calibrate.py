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
from tvm.relay.transform.quantize import Quantizer, QuantizerPattern, QuantizationCalibrator, Conv2DPattern, Conv2DBiasAddPattern, DensePattern, AddPattern, MultiplyPattern, CalibrationCallback
from test_quantize import create_conv2d_func, create_q_conv2d_func, create_conv2d_bias_func, create_q_conv2d_bias_func, create_dense_func, create_q_dense_func, create_add_func, create_q_add_func, create_mul_func, create_q_mul_func
from tvm.relay.frontend.common import infer_type

import numpy as np

# Calls all the methods of CalibrationCallback to make sure they work OK
class TestCalibrationCallback(CalibrationCallback):
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.scale_value = np.array(2).astype('float32')
        self.zp_value = np.array(0.5).astype('int32')

    def calibrate_pattern(self, calibration_info):
        scale_zp_values = {}
        
        for i in range(len(calibration_info.input_scale_zps)):
            scale_name = calibration_info.input_scale_zps[i][0].name_hint
            scale_zp_values[scale_name] = self.scale_value
            zp_name = calibration_info.input_scale_zps[i][1].name_hint
            scale_zp_values[zp_name] = self.zp_value

        inputs = [np.random.rand(*self.input_shape)]

        calibration_info.get_unquantized_layer_inputs(inputs)
        calibration_info.get_unquantized_layer_output(inputs)
        calibration_info.get_quantized_layer_inputs(inputs, scale_zp_values)
        calibration_info.get_quantized_layer_output(inputs, scale_zp_values)

        return scale_zp_values

def test_calibrate(quantizer, quantized_func, params):
    calibrator = QuantizationCalibrator(quantizer)
    calibrated_func = calibrator.calibrate()

    quantized_func = relay.build_module.bind_params_by_name(quantized_func, calibrator.calibration_info.scale_zp_value_map)
    quantized_func = relay.build_module.bind_params_by_name(quantized_func, params)
    quantized_func = infer_type(quantized_func)
    calibrated_func = infer_type(calibrated_func)

    assert tvm.ir.structural_equal(quantized_func, calibrated_func)

def reset_scale_zp_counter():
    # For testing purposes, we reset the static scale counter to zero before calibrating so that our variable names
    # match up properly
    QuantizerPattern.scales_count = 0
    QuantizerPattern.zp_count = 0

def test_conv2d(data_shape, weight_shape, attrs):
    reset_scale_zp_counter()

    conv2d_func, data, weight = create_conv2d_func(data_shape, weight_shape, attrs)
    q_conv2d_func = create_q_conv2d_func(data, weight, weight_shape, attrs)
    
    cc = TestCalibrationCallback(data_shape)
    params = {'weight': np.random.randn(*weight_shape).astype('float32')}
    quantizer = Quantizer(conv2d_func, params, [Conv2DPattern(cc)], skip_first=False, skip_last=False)
    
    test_calibrate(quantizer, q_conv2d_func, params)

def test_conv2d_bias(data_shape, weight_shape, bias_shape, attrs):
    reset_scale_zp_counter()

    conv2d_func, data, weight, bias = create_conv2d_bias_func(data_shape, weight_shape, bias_shape, attrs)
    q_conv2d_func = create_q_conv2d_bias_func(data, weight, bias, weight_shape, attrs)
    
    cc = TestCalibrationCallback(data_shape)
    params = {'weight': np.random.randn(*weight_shape).astype('float32')}
    quantizer = Quantizer(conv2d_func, params, [Conv2DBiasAddPattern(cc)], skip_first=False, skip_last=False)
    
    test_calibrate(quantizer, q_conv2d_func, params)

def test_dense(data_shape, weight_shape, attrs):
    reset_scale_zp_counter()

    dense_func, data, weight = create_dense_func(data_shape, weight_shape, attrs)
    q_dense_func = create_q_dense_func(data, weight, attrs)

    cc = TestCalibrationCallback(data_shape)
    params = {'weight': np.random.randn(*weight_shape).astype('float32')}
    quantizer = Quantizer(dense_func, params, [DensePattern(cc)], skip_first=False, skip_last=False)
    
    test_calibrate(quantizer, q_dense_func, params)

def test_add(lhs_shape, rhs_shape):
    reset_scale_zp_counter()

    add_func, lhs, rhs = create_add_func(lhs_shape, rhs_shape)
    q_add_func = create_q_add_func(lhs, rhs)

    cc = TestCalibrationCallback(lhs_shape)
    params = {'weight': np.random.randn(*rhs_shape).astype('float32')}
    quantizer = Quantizer(add_func, params, [AddPattern(cc)], skip_first=False, skip_last=False)
    
    test_calibrate(quantizer, q_add_func, params)

def test_mul(lhs_shape, rhs_shape):
    reset_scale_zp_counter()

    mul_func, lhs, rhs = create_mul_func(lhs_shape, rhs_shape)
    q_mul_func = create_q_mul_func(lhs, rhs)

    cc = TestCalibrationCallback(lhs_shape)
    params = {'weight': np.random.randn(*rhs_shape).astype('float32')}
    quantizer = Quantizer(mul_func, params, [MultiplyPattern(cc)], skip_first=False, skip_last=False)
    
    test_calibrate(quantizer, q_mul_func, params)

def verify_conv2d():
    test_conv2d((2, 3, 32, 32), (32, 3, 3, 3), {'kernel_size': [3, 3], 'kernel_layout': 'OIHW', 'data_layout': 'NCHW', 'padding': [0, 0, 0, 0]})
    test_conv2d((2, 32, 32, 3), (3, 3, 3, 32), {'kernel_size': [3, 3], 'kernel_layout': 'HWIO', 'data_layout': 'NHWC', 'padding': [0, 0, 0, 0]})

def verify_conv2d_bias():
    test_conv2d_bias((2, 3, 32, 32), (32, 3, 3, 3), (32,), {'kernel_size': [3, 3], 'kernel_layout': 'OIHW', 'data_layout': 'NCHW', 'padding': [0, 0, 0, 0]})
    test_conv2d_bias((2, 32, 32, 3), (3, 3, 3, 32), (32,), {'kernel_size': [3, 3], 'kernel_layout': 'HWIO', 'data_layout': 'NHWC', 'padding': [0, 0, 0, 0]})
    
def verify_dense():
    test_dense((1, 8), (16, 8), {'units': 16})
    test_dense((1, 4), (3, 4), {'units': 3})

def verify_add():
    test_add((1, 2, 3), (1, 2, 3))

def verify_mul():
    test_mul((1, 2, 3), (1, 2, 3))

if __name__ == '__main__':
    verify_conv2d()
    verify_conv2d_bias()
    verify_dense()
    verify_add()
    verify_mul()
