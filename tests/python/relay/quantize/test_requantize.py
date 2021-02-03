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
from tvm.relay.transform.quantize import Requantizer
from tvm.relay.frontend.common import infer_type

import numpy as np

def check_requantize(pre_graph, expected_graph):
    post_graph = Requantizer().requantize(pre_graph)
    print("post_graph before infer type: ", post_graph)

    print("expected_graph before infer type: ", expected_graph)
    post_graph = infer_type(post_graph)
    expected_graph = infer_type(expected_graph)
    print("post_graph: ", post_graph)
    print("expected_graph: ", expected_graph)
    assert tvm.ir.structural_equal(post_graph, expected_graph)


    assert tvm.ir.structural_equal(post_graph, expected_graph)
def make_scale_zp(scale_name, zp_name):
    scale_var = relay.var(scale_name, shape=(), dtype='float32')
    zp_var = relay.var(zp_name, shape=(), dtype='int32')
    return scale_var, zp_var

def test_simple_requantize(data_shape):
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype='int8'))
    scale1, zp1 = relay.const(np.array(1).astype('float32')), relay.const(np.array(1).astype('int32'))

    deq_data = relay.qnn.op.dequantize(int8_data, scale1, zp1)
    scale2, zp2 = relay.const(np.array(2).astype('float32')), relay.const(np.array(2).astype('int32'))
    pre_graph = relay.Function([int8_data], relay.qnn.op.quantize(deq_data, scale2, zp2))

    expected_graph = relay.Function([int8_data], relay.qnn.op.requantize(int8_data, scale1, zp1, scale2, zp2))
    check_requantize(pre_graph, expected_graph)

"""
def test_int8_op_requantize(data_shape):
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype='int8'))
    scale1, zp1 = relay.const(np.array(1).astype('float32')), relay.const(np.array(1).astype('int32'))

    deq_data = relay.qnn.op.dequantize(int8_data, scale1, zp1)
    scale2, zp2 = relay.const(np.array(2).astype('float32')), relay.const(np.array(2).astype('int32'))
    pre_graph = relay.Function([int8_data], relay.qnn.op.quantize(deq_data, scale2, zp2))

    expected_graph = relay.Function([int8_data], relay.qnn.op.requantize(int8_data, scale1, zp1, scale2, zp2))
    
"""
if __name__ == '__main__':
    test_simple_requantize((1, 2, 3, 4))
    

