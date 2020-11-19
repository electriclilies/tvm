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
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, dominates, rewrite

# Dequantize(quantize(var)) -> requantize(var) (when dominated)
# dequantize(int8_op(int8_op(quantize(var)))) -> int8_op(int8_op(requantize(var)))
# int_8_op(int_8_op(dequantize(wildcard()))) -> dequantize(int_8_op(int_8_op(wildcard()))) 
    # This transformation can be done 2nd (perhaps is not a pattern matcher thing, just a depth search..)
class Requantizer():
    class RequantizerCallback(DFPatternCallback):
        def __init__(self):
            super().__init__()
            self.deq_data = wildcard()
            self.dequantize_scale = wildcard()
            self.dequantize_zp = wildcard()
            
            self.q_data = wildcard()
            self.quantize_scale = wildcard()
            self.quantize_zp = wildcard()

            self.dequantize = is_op('qnn.dequantize')(self.deq_data, self.dequantize_scale, self.dequantize_zp)
            self.quantize = is_op('qnn.quantize')(self.q_data, self.quantize_scale, self.quantize_zp)
            self.is_int_8_op = is_op('nn.max_pool2d') | is_op('nn.max_pool3d') | is_op('nn.relu')
            self.pattern = dominates(self.dequantize, self.is_int_8_op, self.quantize)
        
        def callback(self, pre, post, node_map):
            print("callback")
            # How do I get scale and zp out of the pattern?
            deq_data = node_map[self.deq_data][0]
            dequantize_scale = node_map[self.dequantize_scale][0]
            dequantize_zp = node_map[self.dequantize_zp][0]
            print("deq scale: ", dequantize_scale)
            print("deq zp: ", dequantize_zp)
            quantize = node_map[self.quantize][0]
            dequantize = node_map[self.dequantize][0]
            print(self.deq_data)
            print(quantize)
            print(dequantize)

    def requantize(self, mod):
        rewritten_func = rewrite(self.RequantizerCallback(), mod['main'])
        rewritten_mod = tvm.ir.IRModule()
        rewritten_mod['main'] = rewritten_func

        return rewritten_mod


