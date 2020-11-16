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
from tvm.contrib import graph_runtime
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.utils import get_pad_tuple2d
from tvm.relay.dataflow_pattern import rewrite, wildcard, is_op, DFPatternCallback
from collections import OrderedDict

def prerequisite_optimize(mod, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = tvm.transform.Sequential(
        [relay.transform.SimplifyInference(),
         relay.transform.FoldConstant(),
         relay.transform.FoldScaleAxis(),
         relay.transform.CanonicalizeOps(), #TODO: should this be in prereq optimize?
         relay.transform.FoldConstant()])

    if params is not None:
        mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)

    with relay.build_config(opt_level=3):
        mod = optimize(mod)
    return mod

class CalibrationMap:
    def __init__(self, calibration_dict, tuple_subgraph_func, q_tuple_subgraph_func):
        # Map of scale/zp variable names to indices in tuple_subgraph_func output (constructed in QuantizeMutator visit)
        self.calibration_dict = calibration_dict

        # Functions containing tuples of subgraphs that will be evaluated during calibration
        tuple_subgraph_mod = tvm.ir.IRModule()
        tuple_subgraph_mod['main'] = tuple_subgraph_func

        q_tuple_subgraph_mod = tvm.ir.IRModule()
        q_tuple_subgraph_mod['main'] = q_tuple_subgraph_func

        self.tuple_subgraph_mod = tuple_subgraph_mod
        self.q_tuple_subgraph_mod = q_tuple_subgraph_mod

        # TODO: should var map be stored in here for nice packaging?? Perhaps
        self.var_map = []
        

class Quantizer:

    def quantize(self, mod, skip_layers=[0]):
        quantize_mutator = self.QuantizeMutator(skip_layers)
        # SimplifyInference, FoldConstants, FoldScaleAxis
        preoptimized_mod = prerequisite_optimize(mod)
        
        q_fn = quantize_mutator.visit(preoptimized_mod['main'])
        
        out_tuple = relay.Tuple(quantize_mutator.subgraph_list)
        q_out_tuple = relay.Tuple(quantize_mutator.quantized_subgraph_list)
        
        q_fn = relay.Function(list(relay.analysis.free_vars(q_fn)), q_fn)
        
        quantized_mod = tvm.ir.IRModule()
        quantized_mod['main'] = q_fn
        
        out_tuple_fn = relay.Function(list(relay.analysis.free_vars(out_tuple)), out_tuple)
        q_out_tuple_fn = relay.Function(list(relay.analysis.free_vars(q_out_tuple)), q_out_tuple)
        
        calibration_map = CalibrationMap(quantize_mutator.calibration_dict, out_tuple_fn, q_out_tuple_fn)

        return (quantized_mod, calibration_map)
    
    class CalibrationMap:
        def __init__(self, calibration_dict, tuple_subgraph_func, q_tuple_subgraph_func):
            # Map of scale/zp variable names to indices in tuple_subgraph_func output (constructed in QuantizeMutator visit)
            self.calibration_dict = calibration_dict
            
            tuple_subgraph_mod = tvm.ir.IRModule()
            tuple_subgraph_mod['main'] = tuple_subgraph_func

            q_tuple_subgraph_mod = tvm.ir.IRModule()
            q_tuple_subgraph_mod['main'] = q_tuple_subgraph_func

            self.tuple_subgraph_mod = tuple_subgraph_mod
            self.q_tuple_subgraph_mod = q_tuple_subgraph_mod

            # TODO: should var map be stored in here for nice packaging?? Perhaps
            self.var_map = []
        

    class QuantizeMutator(ExprMutator):
        def __init__(self, skip_layers):
            super().__init__()

            # If we are skipping layers, remove duplicates and sort
            self.skip_layers = list(set(skip_layers))
            self.skip_layers.sort()
            self.skip_layers_ptr = 0

            # Number of conv2d and dense we have seen
            self.compute_layer_count = 0
            
            # Counts the number of times we've added a scale and zp for variable naming
            self.scales_count = 0
            self.zp_count = 0

            # Map of variable names to indices subgraph_list and quantized_subgraph_list
            self.calibration_dict = OrderedDict()
            
            # List of subgraphs that will be turned into the tuple output of the Relay
            # function used during calibration. This will let us look at the values
            # of intermediate subgraphs at runtime. 
            self.quantized_subgraph_list = []
            self.subgraph_list = []

        # Helper to construct the relay scale variable for this layer
        def scale(self, name):
            var = relay.var(str(name) + "_scale_" + str(self.scales_count), shape=(), dtype='float32')

            self.scales_count = self.scales_count + 1
            return var

        # Helper to construct the relay zero_point variable for this layer
        def zero_point(self, name):
            var = relay.var(str(name) + "_zero_pt_" + str(self.zp_count), shape=(), dtype='int32')

            self.zp_count = self.zp_count + 1
            return var
        
        # Helper function that adds a subraph to subgraph_list and returns the index it is inserted at
        def add_subgraph(self, subgraph):
            self.subgraph_list.append(subgraph)
            return len(self.subgraph_list) - 1

        # Helper function that adds a subraph to quantized_subgraph_list and returns the index it is inserted at
        def add_quantized_subgraph(self, q_subgraph):
            self.quantized_subgraph_list.append(q_subgraph)
            return len(self.quantized_subgraph_list) - 1

        def visit_call(self, call):
            if call.op == relay.op.get('nn.conv2d'):
                
                pre_data, pre_weight = call.args[0], call.args[1]
                data, weight = self.visit(pre_data), self.visit(pre_weight)

                # Check if we are skipping quantization on this layer after recursive call
                self.compute_layer_count = self.compute_layer_count + 1 # Conv2d is compute layer
                if self.skip_layers and (self.skip_layers_ptr < len(self.skip_layers)):
                    if (self.compute_layer_count == self.skip_layers[self.skip_layers_ptr] + 1):
                        self.skip_layers_ptr = self.skip_layers_ptr + 1
                        return super().visit_call(call)

                # Find kernel_size, channels (will be passed to qnn.conv2d)
                new_attr_dict = {}
                kernel_type_info = infer_type(call.args[1])
                for attr in call.attrs.keys():
                    attr_value = call.attrs[attr]
                    if isinstance(attr_value, tvm.ir.container.Array):
                        attr_value = tuple(attr_value)
                    if attr == 'kernel_size':
                        kernel_size = call.attrs[attr]
                        if kernel_size is None:
                            kernel_size = tuple(kernel_type_info.checked_type.shape[2:4]) # Assumes OIHW layout
                        else:
                            kernel_size = tuple([k.value for k in call.attrs[attr]])
                    elif attr == 'channels':
                        channels = call.attrs[attr]
                        if channels is None:
                            channels = kernel_type_info.checked_type.shape[0] #TODO: Change to work with more layouts
                        channels = channels.value
                    elif attr == 'padding':
                        padding = call.attrs[attr]
                    else:
                        new_attr_dict[str(attr)] = attr_value

                #TODO Figure out if this could be better.
                # Override output dtype.
                new_attr_dict['out_dtype'] = 'int32'

                # Create quantization variables for arguments to this convolution.
                data_scale = self.scale('conv2d_data') 
                data_zp = self.zero_point('conv2d_data')
                weight_scale = self.scale('conv2d_weight')
                weight_zp = self.zero_point('conv2d_weight')
                
                # Quantize the data and construct args for qnn.conv2d
                quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)

                args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale, kernel_size, channels]
                
                if padding is not None:
                    #TODO: Need to make this work with other layouts.
                    top, left, bottom, right = [p.value for p in get_pad_tuple2d(padding)]
                    pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
                    pad_val = 0
                    args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

                # Construct quantized qnn.conv2d and dequantize
                qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
                dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)
                
                pre_data_idx = self.add_subgraph(pre_data)
                quantized_data_idx = self.add_quantized_subgraph(quantized_data)
                pre_weight_idx = self.add_subgraph(pre_weight)
                quantized_weight_idx = self.add_quantized_subgraph(quantized_weight)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)
                
                self.calibration_dict[(data_key, weight_key)] = (((pre_data_idx, quantized_data_idx), (pre_weight_idx, quantized_weight_idx)), (call_idx, dequantized_call_idx))
                
                #TODO: fuse bias_add with conv2d during quantization

                return dequantized_call

            elif call.op == relay.op.get('nn.dense'):
                
                pre_data, pre_weight = call.args[0], call.args[1]
                data, weight = self.visit(pre_data), self.visit(pre_weight)

                # Check if we are skipping quantization on this layer after recursive call
                self.compute_layer_count = self.compute_layer_count + 1 # dense is a compute layer
                if self.skip_layers and (self.skip_layers_ptr < len(self.skip_layers)):
                    if (self.compute_layer_count == self.skip_layers[self.skip_layers_ptr] + 1):
                        self.skip_layers_ptr = self.skip_layers_ptr + 1
                        return super().visit_call(call)

                # Create quantization parameters for arguments to this dense layer.
                data_scale = self.scale('dense_data') 
                data_zp = self.zero_point('dense_data')
                weight_scale = self.scale('dense_weight')
                weight_zp = self.zero_point('dense_weight')
                
                # Quantize data and construct args for qnn.dense
                quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp)

                units = call.attrs['units']
                #print("trying to see if units is noe")
                #print(units == None)
                print("if units")
                # TODO: THIS CAUSES A SEG FAULT
                if units:
                    print("inside")
                    weight_type_info = infer_type(call.args[1])
                    print(weight_type_info)
                    units = weight_type_info.checked_type.shape[0]
                    print(units)
                print("oustide")
                args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale, units]

                qnn_call = relay.qnn.op.dense(*args)
                dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)

                pre_data_idx = self.add_subgraph(pre_data)
                quantized_data_idx = self.add_quantized_subgraph(quantized_data)
                pre_weight_idx = self.add_subgraph(pre_weight)
                quantized_weight_idx = self.add_quantized_subgraph(quantized_weight)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)

                self.calibration_dict[(data_key, weight_key)] = (((pre_data_idx, quantized_data_idx), (pre_weight_idx, quantized_weight_idx)), (call_idx, dequantized_call_idx))

                return dequantized_call

            elif call.op == relay.op.get('add'):
                
                pre_lhs, pre_rhs = call.args[0], call.args[1]
                lhs, rhs = self.visit(pre_lhs), self.visit(pre_rhs)
                
                # Don't quantize the add if it is in a skipped layer
                if self.skip_layers and self.skip_layers_ptr != 0:
                    last_layer = self.compute_layer_count - 1
                    last_skipped = self.skip_layers[self.skip_layers_ptr - 1]
                    if (last_layer == last_skipped):
                        return super().visit_call(call)
                
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

                dequantized_lhs = relay.qnn.op.dequantize(quantized_lhs, lhs_scale, relay.const(0, dtype='int32'))
                dequantized_rhs = relay.qnn.op.dequantize(quantized_rhs, rhs_scale, relay.const(0, dtype='int32'))

                add_scale = relay.op.add(lhs_scale, rhs_scale)

                requantized_lhs = relay.qnn.op.quantize(dequantized_lhs, add_scale, relay.const(0, dtype='int32'))
                requantized_rhs = relay.qnn.op.quantize(dequantized_rhs, add_scale, relay.const(0, dtype='int32'))
        
                add = relay.op.add(requantized_lhs, requantized_rhs)
                dequantized_call = relay.qnn.op.dequantize(add, add_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                lhs_key = (lhs_scale, lhs_zp)
                rhs_key = (rhs_scale, rhs_zp)
                
                pre_lhs_idx = self.add_subgraph(pre_lhs)
                quantized_lhs_idx = self.add_quantized_subgraph(quantized_lhs)
                pre_rhs_idx = self.add_subgraph(pre_rhs)
                quantized_rhs_idx = self.add_quantized_subgraph(quantized_rhs)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)

                self.calibration_dict[(lhs_key, rhs_key)] = (((pre_lhs_idx, quantized_lhs_idx), (pre_rhs_idx, quantized_rhs_idx)), (call_idx, dequantized_call_idx))

                return dequantized_call

            elif call.op == relay.op.get('multiply'):

                pre_lhs, pre_rhs = call.args[0], call.args[1]
                lhs, rhs = self.visit(pre_lhs), self.visit(pre_rhs)

                # Don't quantize multiply if it is in a skipped layer
                if self.skip_layers and self.skip_layers_ptr != 0:
                    last_layer = self.compute_layer_count - 1
                    last_skipped = self.skip_layers[self.skip_layers_ptr - 1]
                    if (last_layer == last_skipped):
                        return super().visit_call(call)

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
                dequantized_call = relay.qnn.op.dequantize(multiply, lhs_scale * rhs_scale, relay.const(0, dtype='int32'))

                # For binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, quantized_a), (b, quantized_b)), (binop_output, quantized_binop_output))
                lhs_key = (lhs_scale, lhs_zp)
                rhs_key = (rhs_scale, rhs_zp)

                pre_lhs_idx = self.add_subgraph(pre_lhs)
                quantized_lhs_idx = self.add_quantized_subgraph(quantized_lhs)
                pre_rhs_idx = self.add_subgraph(pre_rhs)
                quantized_rhs_idx = self.add_quantized_subgraph(quantized_rhs)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)

                self.calibration_dict[(lhs_key, rhs_key)] = (((pre_lhs_idx, quantized_lhs_idx), (pre_rhs_idx, quantized_rhs_idx)), (call_idx, dequantized_call_idx))
                return dequantized_call
            else:
                return super().visit_call(call)
