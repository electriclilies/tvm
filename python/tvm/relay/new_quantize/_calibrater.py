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
from tvm.relay.new_quantize import Quantizer
import numpy as np

class Calibrater():
    def __init__(self, quantizer, target='llvm', ctx=tvm.cpu(), dataset_manager = None): # TODO: is there a better way to deal w target/ctx
        self.quantizer = quantizer

        self.calibration_info = CalibrationInfo(quantizer.tuple_subgraph_mod, quantizer.q_tuple_subgraph_mod, quantizer.partition_infos, dataset_manager, target, ctx)

    def calibrate(self):
        # Create a map of DFPatternCallback to QuantizerPattern
        # What do we need? take quantizer_pattern and partition_info and get the callback function.
        pattern_map = {pattern.pattern:pattern for pattern in self.quantizer.patterns}

        for partition_info in self.calibration_info.partition_infos:
            # Set the partition info so we can access it from the callback
            self.calibration_info.set_current_partition_info(partition_info)
            quantizer_pattern = pattern_map[partition_info.pattern]
            
            # Get the values for scales and ZPs in this layer, store
            scale_zps = quantizer_pattern.calibrate_pattern(self.calibration_info)
            self.calibration_info.update_scale_zp_map(scale_zps)
    
        # TODO: Should this return a mod with params bound in it or just a list of the scales/zps
        calibrated_func = relay.build_module.bind_params_by_name(self.quantizer.q_tuple_subgraph_mod['main'], self.calibration_info.scale_zp_value_map)
        
        # If num_original_outputs is -1, original output wasn't a tuple
        if (self.quantizer.num_original_outputs == -1):
            calibrated_func = relay.Function(calibrated_func.params, calibrated_func.body.fields[0])

        else:
            new_body = relay.Tuple(calibrated_func.body.fields[0:self.quantizer.num_original_outputs])
            calibrated_func = relay.Function(calibrated_func.params, new_body)

            calibrated_func.body = relay.Tuple(calibrated_func.body.fields[0:self.quantizer.num_original_outputs])

        calibrated_mod = tvm.ir.IRModule.from_expr(calibrated_func)

        return calibrated_mod

# Helper class -- rename me?
class CalibrationInfo():
    def __init__(self, tuple_subgraph_mod, q_tuple_subgraph_mod, partition_infos, dataset_manager, target, ctx):
        self.tuple_subgraph_mod = tuple_subgraph_mod
        self.q_tuple_subgraph_mod = q_tuple_subgraph_mod
        self.dataset_manager = dataset_manager
        self.target = target
        self.ctx = ctx

        # Compiled versions of tuple_subgraph_mod and q_tuple_subgraph_mod, created in build_tuple_subgraphs
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]): #TODO: enable AlterOpLayout when fixed
            tuple_subgraph_lib = relay.build(self.tuple_subgraph_mod, target=self.target)
            q_tuple_subgraph_lib = relay.build(self.tuple_subgraph_mod, target=self.target)

        self.tuple_subgraph_graphmodule = graph_runtime.GraphModule(tuple_subgraph_lib["default"](self.ctx))
        self.q_tuple_subgraph_graphmodule = graph_runtime.GraphModule(q_tuple_subgraph_lib["default"](self.ctx))

        self.partition_infos = partition_infos

        self.scale_zp_value_map = {}
        self.initialize_scale_zp_map()

    # Initializes scales to 1, zps to 0. These scales/zps will never be used to calculate
    # intermediate values in the graph returned to the user.
    def initialize_scale_zp_map(self):
        for p_info in self.partition_infos:
            for count in range(len(p_info.input_scale_zps)):
                scale_var = p_info.input_scale_zps[count][0]
                zp_var = p_info.input_scale_zps[count][1]

                self.scale_zp_value_map[scale_var.name_hint] = np.array(1).astype('float32')
                self.scale_zp_value_map[zp_var.name_hint] = np.array(0).astype('int32')
    
    # Sets the partition_info for the current iteration
    def set_current_partition_info(self, partition_info):
        self.partition_info = partition_info

    def update_scale_zp_map(self, new_scale_zps):
        self.scale_zp_value_map.update(new_scale_zps)

    def _run_tuple_mod(self, inputs, idx_list):
        # Set the user provided inputs
        for i, inp in enumerate(inputs):
            self.tuple_subgraph_graphmodule.set_input(i, inp)
        
        # Set the scale and zero points
        self.tuple_subgraph_graphmodule.set_input(**self.scale_zp_value_map)
        self.tuple_subgraph_graphmodule.run()

        value_list = []

        for idx in idx_list:
            value_list.append(self.tuple_subgraph_graphmodule.get_output(idx.value).asnumpy())

        return value_list

    def _run_quantized_tuple_mod(self, inputs, current_layer_scale_zps, idx_list):

        # Set user provided inputs
        for i, inp in enumerate(inputs):
            self.q_tuple_subgraph_graphmodule.set_input(i, inp)
        
        # Set the scale and zero points
        self.q_tuple_subgraph_graphmodule.set_input(**self.scale_zp_value_map)
        self.q_tuple_subgraph_graphmodule.set_input(**current_layer_scale_zps)

        self.q_tuple_subgraph_graphmodule.run()

        value_list = []

        for idx in idx_list:
            value_list.append(self.q_tuple_subgraph_graphmodule.get_output(idx.value).asnumpy())

        return value_list
    
    def get_unquantized_layer_inputs(self, data):
        """Utility function that evaluates the inputs to the current layer and returns the results for given inputs.
        This function should be called from inside _calibration_callback.

        Parameters
        ----------
        inputs : list
            List of inputs to pass into the mod. Inputs appear in the same order they appeared in the original,
            unquantized function.

        Returns
        -------
        input_values : tuple of numpy.ndarray
            A tuple of the values of inputs to the unquantized layer. If the layer is a binop, there will be two elements in the tuple,
            if an n-op, there will be n elements in the tuple.
        """
        return self._run_tuple_mod(data, self.partition_info.input_idxs)

    def get_quantized_layer_inputs(self, data, current_layer_scale_zps):
        """Utility function that evaluates the quantized inputs to the current quantized layer,
        and returns the results in a tuple. It uses previously set scale and zero points when evaluating the graph.
        This function should be called from inside _calibration_callback.

        Parameters
        ----------
        inputs : list
            List of inputs to pass into the mod. Inputs appear in the same order they appeared in the original,
            unquantized function.
        
        current_layer_scale_zps: dictionary
            Map from names of scales and zero points you are setting in the current layer to their values.
            This map should be of the same format as the map you return from _calibration_callback.

        Returns
        -------
        quantized_input_values : tuple of numpy.ndarray
            A tuple of the values of the inputs to the quantized layer. If the layer is a binop, there will be two elements in the tuple,
            if an n-op, there will be n elements in the tuple.
        """
        return self._run_quantized_tuple_mod(data, current_layer_scale_zps, self.partition_info.input_idxs)

    def get_unquantized_layer_output(self, inputs):
        """Utility function that evaluates the unquantized output of the current layer and returns it.
        This function should be called from inside _calibration_callback.

        Parameters
        ----------
        input_list : list
            List of inputs to pass into the mod. Inputs appear in the same order they appeared in the original,
            unquantized function.

        Returns
        -------
        output_value : numpy.ndarray
            The output of this layer.
        """
        return self._run_tuple_mod(inputs, [self.partition_info.output_idx])

    def get_quantized_layer_output(self, data, current_layer_scale_zps):
        """Utility function that evaluates the quantized output of the current layer.
        This function should be called from inside _calibration_callback.

        Parameters
        ----------
        inputs : list
            List of inputs to pass into the mod. Inputs appear in the same order they appeared in the original,
            unquantized function.
        
        current_layer_scale_zps: dictionary
            Map from names of scales and zero points you are setting in the current layer to their values.
            This map should be of the same format as the map you return from _calibration_callback.

        Returns
        -------
        output_value : numpy.ndarray
            The output of the quantized layer.
        """
        return self._run_quantized_tuple_mod(data, current_layer_scale_zps, [self.partition_info.output_idx])
