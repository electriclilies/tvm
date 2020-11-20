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
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, dominates, rewrite

# Dequantize(quantize(var)) -> requantize(var) (when dominated)
# dequantize(int8_op(int8_op(quantize(var)))) -> int8_op(int8_op(requantize(var)))
# int_8_op(int_8_op(dequantize(wildcard()))) -> dequantize(int_8_op(int_8_op(wildcard()))) 
    # This transformation can be done 2nd (perhaps is not a pattern matcher thing, just a depth search..)
class Requantizer():
    # Takes dequantize(is_int8_op*(quantize(data))) -> is_int8_op*(requantize(data))
    class RequantizerCallback(DFPatternCallback):
        def __init__(self):
            super().__init__()
            self.data = wildcard()
            self.dequantize_scale = wildcard()
            self.dequantize_zp = wildcard()
            
            self.quantize_scale = wildcard()
            self.quantize_zp = wildcard()

            self.dequantize = is_op('qnn.dequantize')(self.data, self.dequantize_scale, self.dequantize_zp)
            self.quantize = is_op('qnn.quantize')(wildcard(), self.quantize_scale, self.quantize_zp)
            #self.is_int_8_op = is_op('nn.max_pool2d')| is_op('nn.max_pool3d') | is_op('nn.relu')(wildcard()) | is_op('transpose') | is_op('reshape')
            self.is_int_8_op = is_op('nn.relu')(wildcard())
            self.pattern = dominates(self.dequantize, self.is_int_8_op, self.quantize)

        def callback(self, pre, post, node_map):
            # Extract data from the pattern
            data = node_map[self.data][0]
            dequantize_scale = node_map[self.dequantize_scale][0]
            dequantize_zp = node_map[self.dequantize_zp][0]

            quantize_scale = node_map[self.quantize_scale][0]
            quantize_zp = node_map[self.quantize_zp][0]

            # Rewrite the subgraph using requantize
            if not self.is_int_8_op in node_map:
                res = relay.qnn.op.requantize(data, dequantize_scale, dequantize_zp, quantize_scale, quantize_zp)
            else:
                print("Found case where path is in nodemap, exiting because I don't know what to do with the path yet")
                is_int_8_op = node_map[self.is_int_8_op][0]
                exit()

                #res = relay.qnn.op.requantize(is_int_8_op,  # Todo here? Didnt see this happen in 
            
            return res
    # Takes requantize(quantize(data)) -> quantize(data)
    # TODO: Rename me... 
    class RequantizerCallback2(DFPatternCallback):
        def __init__(self):
            super().__init__()
            
            self.data = wildcard()
            self.output_scale = wildcard()
            self.output_zp = wildcard()

            self.quantize = is_op("qnn.quantize")
            self.requantize = is_op("qnn.requantize")

            self.pattern = self.requantize(self.quantize(self.data, wildcard(), wildcard()), wildcard(), wildcard(), self.output_scale, self.output_zp)

        def callback(self, pre, post, node_map):
            # Extract data from the pattern

            data = node_map[self.data][0]
            output_scale = node_map[self.output_scale][0]
            output_zp = node_map[self.output_zp][0]

            # Rewrite subgraph to just one quantize
            return relay.qnn.op.quantize(data, output_scale, output_zp)
    def requantize(self, mod):
        rewritten_func = rewrite(self.RequantizerCallback(), mod['main'])
        #print("First rewrite done: \n", rewritten_func.astext(False))
        #rewritten_func = rewrite(self.RequantizerCallback2(), rewritten_func)
        rewritten_mod = tvm.ir.IRModule()
        rewritten_mod['main'] = rewritten_func

        return rewritten_mod


