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
import tvm.relay as relay
from tvm.relay.new_quantize import quantize_pass
def test_quantization():
    def check_quantization(op, expected_op):
        print("fjkls")
        original_mod = tvm.ir.IRModule()
        print("fjkdlsjflens")
        original_func = relay.Function((list(relay.analysis.free_vars(op))), op)
        print("fkld")
        original_mod['main'] = original_func

        (quantized_mod, calibration_map) = quantize_pass.quantize(original_mod)
        expected_func = relay.Function((list(relay.analysis.free_vars(expected_op))), op)

        assert tvm.ir.structural_equal(quantized_mod['main'], expected_func, map_free_vars=True)
        # TODO: check stuff about calibration map
    
    # check_conv2d
    """
            data_shape = (2, 16, 16, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 3, 4, 1)
        kernel_dtype = "uint8"
    """

    data = relay.var('data', shape=(1, 64, 56, 56), dtype='float32')
    weight = relay.var('weight', shape=(64, 3, 3, 64), dtype='float32')
    conv2d = relay.op.nn.conv2d(data, weight)

    data_scale, data_zp = relay.var('data_scale'), relay.var('data_zp')
    weight_scale, weight_zp = relay.var('weight_scale'), relay.var('weight_zp')
    q_data = relay.qnn.op.quantize(data, data_scale, data_zp)
    q_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)

    print("kjfdls")
    q_conv2d = relay.qnn.op.conv2d(q_data, q_weight, data_zp, weight_zp, data_scale, weight_scale, channels=64, kernel_size=relay.const((3, 3), dtype='int32'))
    deq_conv2d = relay.qnn.op.dequantize(q_conv2d, data_scale * weight_scale, relay.const(0, dtype='int32'))
    
    check_quantization(conv2d, deq_conv2d)

    # check conv2d with padding

test_quantization()