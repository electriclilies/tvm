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
from tvm import te
from tvm.contrib import graph_runtime
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type
from tvm.relay.new_quantize import CalibrationMap
from tvm.relay.op.nn.utils import get_pad_tuple2d
from collections import OrderedDict

class Quantizer:

    def quantize(self, mod, params, skip_layers=[0], skip_ops=[]):
        """Inserts quantize and dequantize around every layer in the graph

        Parameters
        ----------
        mod : tvm.IRModule
            The module to quantize. We expect that it contains one function called 'main'
		
        params : dict of str to NDArray
            Input parameters to the graph that will be constant folded in prior to quantization.

		skip_layers : list
            A list of the layers we will not quantize, by index.
            E.G., if skip_layers=[0], the first layer will not be quantized.
            In this case, a layer is a conv2d or dense node in the graph, along
            with its inputs and any transformations done to the outputs.
 
        Returns
        -------
        quantized_mod : tvm.IRModule
            The quantized module, with scale and zero points inserted as relay variables.
            These variables are provided as parameters to the main function
        
        calibration_map : tvm.relay.quantize.CalibrationMap
            Helper class containing info gathered by the Quantizer that will
            be used by the Calibrater
        
        """
        quantize_mutator = self.QuantizeMutator(skip_layers, skip_ops)

        # SimplifyInference, FoldConstants, FoldScaleAxis
        preoptimized_mod = prerequisite_optimize(mod, params)
        
        q_fn = quantize_mutator.visit(preoptimized_mod['main'])

        # Make sure that inputs to quantized function appear in correct order
        original_inputs = list(relay.analysis.free_vars(preoptimized_mod['main'].body))
        scale_zp_inputs = [var for var in list(relay.analysis.free_vars(q_fn.body)) if var not in original_inputs]
        q_inputs = original_inputs + scale_zp_inputs
        q_fn = relay.Function(q_inputs, q_fn.body)
        
        quantized_mod = tvm.ir.IRModule()
        quantized_mod['main'] = q_fn
    
        out_tuple = relay.Tuple(quantize_mutator.subgraph_list)
        q_out_tuple = relay.Tuple(quantize_mutator.quantized_subgraph_list)

        out_tuple_fn = relay.Function(list(relay.analysis.free_vars(out_tuple)), out_tuple)
        q_out_tuple_fn = relay.Function(list(relay.analysis.free_vars(q_out_tuple)), q_out_tuple)
        
        calibration_map = CalibrationMap(quantize_mutator.output_index_map, out_tuple_fn, q_out_tuple_fn)
        return (quantized_mod, calibration_map)
        

    class QuantizeMutator(ExprMutator):
        def __init__(self, skip_layers, skip_ops):
            super().__init__()

            # If we are skipping layers, remove duplicates and sort
            self.skip_layers = list(set(skip_layers))
            self.skip_layers.sort()
            self.skip_layers_ptr = 0

            self.skip_ops = skip_ops

            # Number of conv2d and dense we have seen
            self.compute_layer_count = 0
            
            # Counts the number of times we've added a scale and zp for variable naming
            self.scales_count = 0
            self.zp_count = 0

            # Map of variable names to indices subgraph_list and quantized_subgraph_list
            self.output_index_map = OrderedDict()
            
            # List of subgraphs that will be turned into the tuple output of the Relay
            # function used during calibration. This will let us look at the values
            # of intermediate subgraphs at runtime. 
            self.quantized_subgraph_list = []
            self.subgraph_list = []

        # Helper to construct the relay scale variable for this layer
        def scale(self, name):
            #var = relay.var(str(name) + "_scale_" + str(self.scales_count), shape=(te.size_var("channels"),), dtype='float32')
            var = relay.var(str(name) + "_scale_" + str(self.scales_count), shape=(), dtype='float32')
            self.scales_count = self.scales_count + 1
            return var

        # Helper to construct the relay zero_point variable for this layer
        def zero_point(self, name):
            #var = relay.var(str(name) + "_zero_pt_" + str(self.zp_count), shape=(te.size_var("channels"),), dtype='int32')
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
                
                # Check whether skipping nn.conv
                if relay.op.get('nn.conv2d') in self.skip_ops:
                    return super().visit_call(call)

                # Visit call inputs
                pre_data, pre_weight = call.args[0], call.args[1]
                data, weight = self.visit(pre_data), self.visit(pre_weight)
                out_dtype = infer_type(call).checked_type.dtype

                # Check if we are skipping quantization on this layer after recursive call
                self.compute_layer_count = self.compute_layer_count + 1 # Conv2d is compute layer
                if self.skip_layers and (self.skip_layers_ptr < len(self.skip_layers)):
                    if (self.compute_layer_count == self.skip_layers[self.skip_layers_ptr] + 1):
                        self.skip_layers_ptr = self.skip_layers_ptr + 1
                        return super().visit_call(call)

                # Find kernel_size, channels (will be passed to qnn.conv2d)
                new_attr_dict = {}
                kernel_type_info = infer_type(call.args[1])
                kernel_layout = call.attrs["kernel_layout"]
                data_layout = call.attrs["data_layout"]
                
                if kernel_layout == "OIHW":
                    weight_channel_axis = 0
                elif kernel_layout == "HWIO":
                    weight_channel_axis = 3
                else:
                    raise ValueError("Quantizing kernel layout %s for conv2d is not yet supported. Please use OIHW or HWIO", kernel_layout)

                if data_layout == "NCHW":
                    data_channel_axis = 1
                elif data_layout == "NHWC":
                    data_channel_axis = 3
                else:
                    raise ValueError("Quantizing data layout %s for conv2d is not yet supported. Please use NCHW or NHWC", data_layout)

                for attr in call.attrs.keys():
                    attr_value = call.attrs[attr]
                    if isinstance(attr_value, tvm.ir.container.Array):
                        attr_value = tuple(attr_value)
                    if attr == 'kernel_size':
                        kernel_size = call.attrs[attr]
                        if kernel_size is None:
                            if kernel_layout == "OIHW":
                                kernel_size = tuple(kernel_type_info.checked_type.shape[2:4])
                            elif kernel_layout == "HWIO":
                                kernel_size = tuple(kernel_type_info.checked_type.shape[0:2])
                            else:
                                raise ValueError("Quantizting kernel layout %s for conv2d is not yet supported. Please use OIHW or HWIO", kernel_layout)
                        else:
                            kernel_size = tuple([k.value for k in call.attrs[attr]])
                        new_attr_dict[attr] = kernel_size
                    elif attr == 'channels':
                        channels = call.attrs[attr]
                        if channels is None:
                            channels = kernel_type_info.checked_type.shape[weight_channel_axis].value
                        new_attr_dict[attr] = channels
                    elif attr == 'padding':
                        padding = call.attrs[attr] # Don't need to put padding in attr dict because we explicitly construct padding
                    else:
                        new_attr_dict[attr] = attr_value

                new_attr_dict['out_dtype'] = 'int32'

                # Create quantization variables for arguments to this convolution.
                data_scale = self.scale('conv2d_data') 
                data_zp = self.zero_point('conv2d_data')
                weight_scale = self.scale('conv2d_weight')
                weight_zp = self.zero_point('conv2d_weight')
                
                # Quantize the data and construct args for qnn.conv2d
                quantized_data = relay.qnn.op.quantize(data, data_scale, data_zp)
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp, axis=weight_channel_axis)

                args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]
                
                if padding is not None:
                    #TODO: Need to make this work with other layouts.
                    top, left, bottom, right = [p.value for p in get_pad_tuple2d(padding)]
                    pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
                    pad_val = 0
                    args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

                # Construct quantized qnn.conv2d and dequantize
                qnn_call = relay.qnn.op.conv2d(*args, **new_attr_dict)
                dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype, axis=data_channel_axis)

                # For binop_output = binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, b), (quantized_a, quantized_b)), (binop_output, quantized_binop_output), (relay_op, op_attributes))
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)
                
                pre_data_idx = self.add_subgraph(pre_data)
                quantized_data_idx = self.add_quantized_subgraph(quantized_data)
                pre_weight_idx = self.add_subgraph(pre_weight)
                quantized_weight_idx = self.add_quantized_subgraph(quantized_weight)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)
                
                self.output_index_map[(data_key, weight_key)] = (((pre_data_idx, pre_weight_idx), (quantized_data_idx, quantized_weight_idx)), (call_idx, dequantized_call_idx), (relay.op.get('qnn.conv2d'), new_attr_dict))
                #TODO: fuse bias_add with conv2d during quantization.. not sure how this 
                # Maybe make dense a fn and then if we find a dense in an add, call the fn

                return dequantized_call

            elif call.op == relay.op.get('nn.dense'):

                # Check whether skipping nn.dense
                if relay.op.get('nn.dense') in self.skip_ops:
                    return super().visit_call(call)

                # Visit call inputs
                pre_data, pre_weight = call.args[0], call.args[1]
                data, weight = self.visit(pre_data), self.visit(pre_weight)
                out_dtype = infer_type(call).checked_type.dtype

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
                quantized_weight = relay.qnn.op.quantize(weight, weight_scale, weight_zp, axis=0) # Axis = 0 for per channel quantization

                args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]

                new_attr_dict = {}
                units = call.attrs['units']
                if units is None:
                    weight_type_info = infer_type(call.args[1])
                    units = weight_type_info.checked_type.shape[0]
                units = units.value
                new_attr_dict['units'] = units

                qnn_call = relay.qnn.op.dense(*args, **new_attr_dict)
                dequantized_call = relay.qnn.op.dequantize(qnn_call, data_scale * weight_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype, axis=1) # Dequantize axis = 1

                # For binop_output = binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, b), (quantized_a, quantized_b)), (binop_output, quantized_binop_output), (relay_op, op_attributes))
                data_key = (data_scale, data_zp)
                weight_key = (weight_scale, weight_zp)

                pre_data_idx = self.add_subgraph(pre_data)
                quantized_data_idx = self.add_quantized_subgraph(quantized_data)
                pre_weight_idx = self.add_subgraph(pre_weight)
                quantized_weight_idx = self.add_quantized_subgraph(quantized_weight)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)

                self.output_index_map[(data_key, weight_key)] = (((pre_data_idx, pre_weight_idx), (quantized_data_idx, quantized_weight_idx)), (call_idx, dequantized_call_idx), (relay.op.get('qnn.dense'), new_attr_dict))

                return dequantized_call
            elif call.op == relay.op.get('add'):

                # Check whether skipping add
                if relay.op.get('nn.conv2d') in self.skip_ops:
                    return super().visit_call(call)

                # Visit call inputs
                pre_lhs, pre_rhs = call.args[0], call.args[1]
                lhs, rhs = self.visit(pre_lhs), self.visit(pre_rhs)
                out_dtype = infer_type(call).checked_type.dtype
                
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

                dequantized_lhs = relay.qnn.op.dequantize(quantized_lhs, lhs_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)
                dequantized_rhs = relay.qnn.op.dequantize(quantized_rhs, rhs_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)

                add_scale = relay.op.add(lhs_scale, rhs_scale)

                requantized_lhs = relay.qnn.op.quantize(dequantized_lhs, add_scale, relay.const(0, dtype='int32'))
                requantized_rhs = relay.qnn.op.quantize(dequantized_rhs, add_scale, relay.const(0, dtype='int32'))
        
                add = relay.op.add(requantized_lhs, requantized_rhs)
                dequantized_call = relay.qnn.op.dequantize(add, add_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)

                # For binop_output = binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, b), (quantized_a, quantized_b)), (binop_output, quantized_binop_output), (relay_op, op_attributes))
                lhs_key = (lhs_scale, lhs_zp)
                rhs_key = (rhs_scale, rhs_zp)
                
                pre_lhs_idx = self.add_subgraph(pre_lhs)
                quantized_lhs_idx = self.add_quantized_subgraph(quantized_lhs)
                pre_rhs_idx = self.add_subgraph(pre_rhs)
                quantized_rhs_idx = self.add_quantized_subgraph(quantized_rhs)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)

                self.output_index_map[(lhs_key, rhs_key)] = (((pre_lhs_idx, pre_rhs_idx), (quantized_lhs_idx, quantized_rhs_idx)), (call_idx, dequantized_call_idx), (relay.op.get('add'), {}))

                return dequantized_call

            elif call.op == relay.op.get('multiply'):

                # Check whether skipping multiply
                if relay.op.get('multiply') in self.skip_ops:
                    return super().visit_call(call)

                # Visit call inputs
                pre_lhs, pre_rhs = call.args[0], call.args[1]
                lhs, rhs = self.visit(pre_lhs), self.visit(pre_rhs)
                out_dtype = infer_type(call).checked_type.dtype
                
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
                dequantized_call = relay.qnn.op.dequantize(multiply, lhs_scale * rhs_scale, relay.const(0, dtype='int32'), out_dtype=out_dtype)

                # For binop_output = binop(a, b), we map ((scale_a, zero_point_a), (scale_b, zero_point_b)) to (((a, b), (quantized_a, quantized_b)), (binop_output, quantized_binop_output), (relay_op, op_attributes))
                lhs_key = (lhs_scale, lhs_zp)
                rhs_key = (rhs_scale, rhs_zp)

                pre_lhs_idx = self.add_subgraph(pre_lhs)
                quantized_lhs_idx = self.add_quantized_subgraph(quantized_lhs)
                pre_rhs_idx = self.add_subgraph(pre_rhs)
                quantized_rhs_idx = self.add_quantized_subgraph(quantized_rhs)
                call_idx = self.add_subgraph(call)
                dequantized_call_idx = self.add_quantized_subgraph(dequantized_call)

                self.output_index_map[(lhs_key, rhs_key)] = (((pre_lhs_idx, pre_rhs_idx), (quantized_lhs_idx, quantized_rhs_idx)), (call_idx, dequantized_call_idx), (relay.op.get("add"), {}))

                return dequantized_call
            
            else:
                return super().visit_call(call)

def prerequisite_optimize(mod, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = tvm.transform.Sequential(
        [relay.transform.DynamicToStatic(),
         relay.transform.SimplifyInference(),
         relay.transform.FoldConstant(),
         relay.transform.FoldScaleAxis(),
         relay.transform.CanonicalizeOps(), #TODO: should this be in prereq optimize?
         relay.transform.FoldConstant()])

    if params is not None:
        mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)

    with relay.build_config(opt_level=3):
        mod = optimize(mod)
    return mod