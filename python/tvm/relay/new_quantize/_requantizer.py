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

# Dequantize(quantize(var)) -> requantize(var)
# dequantize(int8_op(int8_op(quantize(var)))) -> int8_op(int8_op(requantize(var)))
class Requantizer():
    # Ops that are permitted inbetween quantize and dequantize if we are rewriting to requantize
    is_int_8_op = is_op('nn.max_pool2d')(wildcard()) | is_op('nn.max_pool3d')(wildcard()) | is_op('nn.relu')(wildcard()) | is_op('transpose')(wildcard()) | is_op('reshape')(wildcard()) | is_op('nn.pad')(wildcard()) | is_op('squeeze')(wildcard())
    # Takes dequantize(is_int8_op*(quantize(data))) -> requantize(is_int8_op*(data))

    class RequantizerCallback(DFPatternCallback):
        def __init__(self):
            super().__init__()
            self.data = wildcard()
            self.dequantize_scale = wildcard()
            self.dequantize_zp = wildcard()
            
            self.quantize_scale = wildcard()
            self.quantize_zp = wildcard()

            # Ops that are permitted inbetween quantize and dequantize if we are rewriting to requantize
            self.is_int_8_op_func = lambda x: (is_op('nn.max_pool2d')(x) | is_op('nn.max_pool2d')(x) | is_op('nn.max_pool3d')(x) | is_op('nn.relu')(x) | is_op('transpose')(x) | is_op('reshape')(x) | is_op('nn.pad')(x) | is_op('squeeze')(x))
            self.is_int_8_op = is_op('nn.max_pool2d')(wildcard()) | is_op('nn.max_pool2d')(wildcard()) | is_op('nn.max_pool3d')(wildcard()) | is_op('nn.relu')(wildcard()) | is_op('transpose')(wildcard()) | is_op('reshape')(wildcard()) | is_op('nn.pad')(wildcard()) | is_op('squeeze')(wildcard())

            #self.is_int_8_op = self.is_int_8_op_func(wildcard())

            # Pattern allowing quantize(is_int_8_op*(dequantize(data))) -- (with 1 or more is_int_8_ops)
            self.dequantize = is_op('qnn.dequantize')(self.data, self.dequantize_scale, self.dequantize_zp)

            self.dominator = dominates(self.dequantize, self.is_int_8_op, self.is_int_8_op)
            self.quantize = is_op('qnn.quantize')(self.dominator, self.quantize_scale, self.quantize_zp)

            # For resnets, dominator pattern above does not work, so we add this pattern (which allows a max of 5 nodes inbetween the quantize and dequantize)
            self.resnet_dequantize = is_op('qnn.dequantize')(self.data, self.dequantize_scale, self.dequantize_zp).optional(self.is_int_8_op_func)#.optional(self.is_int_8_op_func).optional(self.is_int_8_op_func).optional(self.is_int_8_op_func).optional(self.is_int_8_op_func)
            self.resnet_quantize = is_op('qnn.quantize')(self.resnet_dequantize, self.quantize_scale, self.quantize_zp)

            # Pattern with the null path -- quantize(dequantize(data)) -- (no is_int_8_op inbetween)
            # We have to do the null path outside the dominator pattern because of pattern matcher limitations
            self.no_path_dequantize = is_op('qnn.dequantize')(self.data, self.dequantize_scale, self.dequantize_zp)
            self.no_path_quantize = is_op('qnn.quantize')(self.no_path_dequantize, self.quantize_scale, self.quantize_zp)

            self.pattern = self.resnet_quantize
            #self.pattern = self.quantize | self.resnet_quantize | self.no_path_quantize


        def callback(self, pre, post, node_map):
            # Extract data from the pattern
            data = node_map[self.data][0]
            dequantize_scale = node_map[self.dequantize_scale][0]
            dequantize_zp = node_map[self.dequantize_zp][0]

            quantize_scale = node_map[self.quantize_scale][0]
            quantize_zp = node_map[self.quantize_zp][0]

            # Case where there are no ops in between the dequantize and quantize
            if self.no_path_quantize in node_map:
                res = relay.qnn.op.requantize(data, dequantize_scale, dequantize_zp, quantize_scale, quantize_zp)
            # Ops inbetween quantize and dequantize are dominated
            elif self.quantize in node_map:
                for key, value in node_map.items():
                    print(key)
                print(node_map[self.is_int_8_op_func])

                # There are ops in between the dequantize and quantize
                # Takes dequantize(is_int8_op*(quantize(data))) -> requantize(is_int8_op*(data))
                transformed_data = data
                for i in range(len(node_map[self.is_int_8_op]) - 1, -1, -1):
                    call = node_map[self.is_int_8_op][i]
                    # Transform relu into max(zeropoint)
                    if call.op == relay.op.get('nn.relu'):
                        # TODO: Accuracy seemed to go up when using max instead of relu... why??
                        if dequantize_zp.data.asnumpy() == relay.const(0, dtype='int32').data.asnumpy():
                            transformed_data = relay.op.nn.relu(transformed_data)
                        else:
                            transformed_data = relay.op.maximum(transformed_data, relay.cast(dequantize_zp, 'int8'))
                # How do I copy named args (ie attrs) (call.attrs)
                    elif call.op == relay.op.get('nn.max_pool2d'):
                        transformed_data = relay.op.nn.max_pool2d(transformed_data, **call.attrs)
                    elif call.op == relay.op.get('nn.max_pool3d'):
                        transformed_data = relay.op.nn.max_pool3d(transformed_data, **call.attrs)
                    elif call.op == relay.op.get('transpose'):
                        transformed_data = relay.op.transpose(transformed_data, **call.attrs)
                    elif call.op == relay.op.get('reshape'):
                        # For some reason reverse is set as an attr (for topi?), but is not an argument to reshape
                        assert call.attrs['reverse'] == 0
                        attrs = call.attrs.copy()
                        attrs.pop('reverse')
                        transformed_data = relay.op.reshape(transformed_data, **attrs)
                    elif call.op == relay.op.get('nn.pad'):
                        transformed_data = relay.op.nn.pad(transformed_data, **call.attrs)
                    elif call.op == relay.op.get('squeeze'):
                        transformed_data = relay.op.squeeze(transformed_data, **call.attrs)
                    else:
                        # TODO: turn into internal error message
                        assert False, "Uh oh, you must have missed converting an op that is in is_int_8_op"

                res = relay.qnn.op.requantize(transformed_data, dequantize_scale, dequantize_zp, quantize_scale, quantize_zp)
            # Ops inbetween quantize and dequantize are not dominated (this pattern usually seen in resnets, etc)
            else:
                # THIS IS WHERE THE FUNNY BUSINESS IS HAPPENING!
                print("found one!!")
                for key, value in node_map.items():
                    print("KEY: ")
                    print(key)
                
                print("keys printed")
                print(node_map[self.is_int_8_op_func])
                print("successfully passed function into node_map")
                exit()
                return pre

            return res


    class RequantizeChainCallback(DFPatternCallback):
        # Takes a chain of requantizes and turns them into one requantize
        def __init__(self):
            super().__init__()
            self.data = wildcard()
            self.rq_parent_scale = wildcard()
            self.rq_parent_zp = wildcard()

            self.rq_child_scale = wildcard()
            self.rq_child_zp = wildcard()

            self.rq_parent = is_op('qnn.requantize')(self.data, self.rq_parent_scale, self.rq_parent_zp, wildcard(), wildcard())
            self.rq_child = is_op('qnn.requantize')(wildcard(), wildcard(), wildcard(), self.rq_child_scale, self.rq_child_zp)

            self.pattern = dominates(self.rq_parent, self.rq_child, self.rq_child)

        def callback(self, pre, post, node_map):
            data = node_map[self.data][0]
            rq_parent_scale = node_map[self.rq_parent_scale][0]
            rq_parent_zp = node_map[self.rq_parent_zp][0]

            len_child_scales = len(node_map[self.rq_child_scale])
            rq_child_scale = node_map[self.rq_child_scale][len_child_scales-1]

            len_child_zps = len(node_map[self.rq_child_zp])
            rq_child_zp = node_map[self.rq_child_zp][len_child_zps-1]

            return relay.qnn.op.requantize(data, rq_parent_scale, rq_parent_zp, rq_child_scale, rq_child_zp)


    # Takes requantize(quantize(data, scale, zp), rscale, rzp) -> quantize(data, rscale, rzp)
    # TODO: Rename me...
    class ConsolidateRequantizeandQuantize(DFPatternCallback):
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
    
    # Is it worth moving dequantizes as far down as possible so most things are in int8? Would be p easy to add.
    def requantize(self, mod):
        print("Calibrated mod: \n", mod.astext(False))
        #mod = self._split_resnet_dequantize(mod)
        rewritten_func = rewrite(self.RequantizerCallback(), mod['main'])
        print("First rewrite done: \n", rewritten_func.astext(False))
        rewritten_func = rewrite(self.RequantizeChainCallback(), rewritten_func)
        print("Second rewrite done: \n", rewritten_func.astext(False))
        rewritten_func = rewrite(self.ConsolidateRequantizeandQuantize(), rewritten_func)
        print("Third rewrite done: \n", rewritten_func.astext(False))
        #rewritten_func = rewrite(self.RequantizerCallback(), rewritten_func)
        rewritten_mod = tvm.ir.IRModule()
        rewritten_mod['main'] = rewritten_func

        return rewritten_mod


