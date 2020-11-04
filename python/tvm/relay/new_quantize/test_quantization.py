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
import numpy as np
import tvm
import tvm.relay as relay
import tvm.testing
from tvm.relay.new_quantize import quantize_pass
from tvm.relay.frontend.common import infer_type
from tvm.contrib import graph_runtime
from tvm.relay.new_quantize import GlobalCalibrater

# TODO: how to do target
def check_quantization(op, expected_op):

    original_mod = tvm.ir.IRModule()
    original_func = relay.Function((list(relay.analysis.free_vars(op))), op)
    original_mod['main'] = original_func

    (quantized_mod, calibration_map) = quantize_pass.quantize(original_mod, target='llvm', ctx=tvm.cpu())

    expected_func = relay.Function((list(relay.analysis.free_vars(expected_op))), expected_op)
    tvm.ir.assert_structural_equal(quantized_mod['main'].body, expected_func.body, map_free_vars=True)

# Checks that the calibration_map is constructed correctly
# Note: normally the user should not use calibration_map directly. Please write a calibration_callback instead
def check_calibration_map(calibration_map, inputs):
    for (variable_pairs), (input_subgraph_pairs, output_subgraph_pair) in calibration_map.items():
        output_subgraph_fn = output_subgraph_pair[0]
        quantized_output_subgraph_fn = output_subgraph_pair[1]
        assert isinstance(output_subgraph_fn, relay.Function)
        assert isinstance(quantized_output_subgraph_fn, relay.Function)
        output_params = output_subgraph_pair[0].params
        q_output_params = output_subgraph_pair[1].params
        for ((scale, zp), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(variable_pairs, input_subgraph_pairs):
            assert isinstance(scale, relay.Var)
            assert isinstance(zp, relay.Var)
            assert isinstance(data_subgraph_fn, relay.Function)
            assert isinstance(quantized_data_subgraph_fn, relay.Function)

            data_params = data_subgraph_fn.params
            q_data_params = quantized_data_subgraph_fn.params

            # Scale, zp vars should not be in unquantized subgraphs
            assert scale not in data_params
            assert zp not in data_params
            assert scale not in output_params
            assert zp not in output_params

            # Scale, zp vars should be in quantized subgraphs
            assert scale in q_data_params
            assert zp in q_data_params
            assert scale in q_output_params
            assert zp in q_output_params

# Test that conv2d is transformed to qnn.conv2d correctly and that calibration_map is correct
def test_conv2d():

    data = relay.var('data', shape=(1, 3, 224, 224), dtype='float32')
    weight = relay.var('weight', shape=(64, 3, 7, 7), dtype='float32')
    conv2d = relay.op.nn.conv2d(data, weight, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])

    data_scale, data_zp = relay.var('conv2d_data_scale_0', shape=(), dtype='float32'), relay.var('conv2d_data_zero_pt_0', shape=(), dtype='int32')
    weight_scale, weight_zp = relay.var('conv2d_weight_scale_1', shape=(), dtype='float32'), relay.var('conv2d_weight_zero_pt_1', shape=(), dtype='int32')
    q_data = relay.qnn.op.quantize(data, data_scale, data_zp)
    q_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)
    padded_q_data = relay.nn.pad(q_data, pad_width=[[0, 0], [0, 0], [3, 3], [3, 3]])
    q_conv2d = relay.qnn.op.conv2d(padded_q_data, q_weight, data_zp, weight_zp, data_scale, weight_scale, channels=64, strides=[2, 2], kernel_size=[7, 7])
    deq_conv2d = relay.qnn.op.dequantize(q_conv2d, data_scale * weight_scale, relay.const(0, dtype='int32'))
    
    check_quantization(conv2d, deq_conv2d)

# Test that dense is transformed to qnn.dense correctly and that calibration_map is correct
def test_dense():
    data = relay.var('data', shape=(16, 8))
    dense_weight = relay.var('dense_weight', shape=(8, 8))
    units = 8
    dense = relay.nn.dense(data, weight=dense_weight, units=units)

    data_scale, data_zp = relay.var('dense_data_scale_0', shape=(), dtype='float32'), relay.var('dense_data_zero_pt_0', shape=(), dtype='int32')
    weight_scale, weight_zp = relay.var('dense_weight_scale_1', shape=(), dtype='float32'), relay.var('dense_weight_zero_pt_1', shape=(), dtype='int32')
    q_data = relay.qnn.op.quantize(data, data_scale, data_zp)
    q_weight = relay.qnn.op.quantize(dense_weight, weight_scale, weight_zp)
    q_dense = relay.qnn.op.dense(q_data, q_weight, data_zp, weight_zp, data_scale, weight_scale, units)
    deq_dense = relay.qnn.op.dequantize(q_dense, data_scale * weight_scale, relay.const(0, dtype='int32'))

    check_quantization(dense, deq_dense)
    

# Test that add is transformed to qnn.add correctly and that calibration_map is correct
def test_add():
    lhs = relay.var('lhs', shape=(1, 2, 3))
    rhs = relay.var('rhs', shape=(1, 2, 3))
    add = relay.add(lhs, rhs)

    lhs_scale, lhs_zp = relay.var('lhs_scale', shape=(), dtype='float32'), relay.var('lhs_zp', shape=(), dtype='int32')
    rhs_scale, rhs_zp = relay.var('rhs_scale', shape=(), dtype='float32'), relay.var('rhs_zp', shape=(), dtype='int32')
    q_lhs = relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp)
    q_rhs = relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)
    q_add = relay.qnn.op.add(q_lhs, q_rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, lhs_scale + rhs_scale, relay.const(0, dtype='int32'))
    deq_add = relay.qnn.op.dequantize(q_add, lhs_scale + rhs_scale, relay.const(0, dtype='int32'))

    check_quantization(add, deq_add)

# Test that multiply is transformed to qnn.mul correctly and that calibration_map is correct
def test_mul():
    lhs = relay.var('lhs', shape=(1, 2, 3))
    rhs = relay.var('rhs', shape=(1, 2, 3))
    mul = relay.multiply(lhs, rhs)

    lhs_scale, lhs_zp = relay.var('lhs_scale', shape=(), dtype='float32'), relay.var('lhs_zp', shape=(), dtype='int32')
    rhs_scale, rhs_zp = relay.var('rhs_scale', shape=(), dtype='float32'), relay.var('rhs_zp', shape=(), dtype='int32')
    q_lhs = relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp)
    q_rhs = relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)
    q_mul = relay.qnn.op.mul(q_lhs, q_rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, lhs_scale * rhs_scale, relay.const(0, dtype='int32'))
    deq_mul = relay.qnn.op.dequantize(q_mul, lhs_scale * rhs_scale, relay.const(0, dtype='int32'))

    check_quantization(mul, deq_mul)

def get_tvm_output(mod, inputs, target='llvm', ctx=tvm.cpu()):
    with relay.build_config(opt_level=3):
        lib = relay.build(mod, target=target)

    gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    gmod.set_input(**inputs)
    gmod.run()
    return gmod.get_output(0).asnumpy()

# Check that globally quantized conv2d produces reasonable output
def verify_conv_output(data_shape, weight_shape):
    data = relay.var('data', shape=data_shape, dtype='float32')
    weight = relay.var('weight', shape=weight_shape, dtype='float32')
    conv2d = relay.op.nn.conv2d(data, weight, strides=[1, 1], padding=([int(np.ceil(weight_shape[-1] / 2) - 1)] * len(data_shape)), channels=weight_shape[0], kernel_size=weight_shape[2:])

    data_np = np.random.randn(*data_shape).astype('float32')
    weight_np = np.random.randn(*weight_shape).astype('float32')

    original_mod = tvm.ir.IRModule.from_expr(conv2d)

    (quantized_mod, calibration_map) = quantize_pass.quantize(original_mod, target='llvm', ctx=tvm.cpu())

    global_calibrater = GlobalCalibrater(0.05, 0, 0.05, 0)
    calibrated_mod = global_calibrater.calibrate(quantized_mod, calibration_map)
    inputs = {'data': data_np, 'weight': weight_np}

    out = get_tvm_output(calibrated_mod, inputs)
    q_out = get_tvm_output(original_mod, inputs)
    tvm.testing.assert_allclose(out, q_out, atol=0.5)

    check_calibration_map(calibration_map, inputs)

def test_conv_output():
    verify_conv_output([1, 1, 3, 3], [4, 1, 3, 3])

if __name__ == "__main__":
    test_conv2d()
    test_dense()
    test_add()
    test_mul()
    test_conv_output()
    