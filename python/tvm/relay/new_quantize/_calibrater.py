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

import numpy as np

class CalibrationMap:
    def __init__(self, output_index_map, tuple_subgraph_func, q_tuple_subgraph_func):
        # Map of scale/zp variable names to indices in tuple_subgraph_func output (constructed in QuantizeMutator visit)
        self.output_index_map = output_index_map
        
        tuple_subgraph_mod = tvm.ir.IRModule()
        tuple_subgraph_mod['main'] = tuple_subgraph_func

        q_tuple_subgraph_mod = tvm.ir.IRModule()
        q_tuple_subgraph_mod['main'] = q_tuple_subgraph_func

        self.tuple_subgraph_mod = tuple_subgraph_mod
        self.q_tuple_subgraph_mod = q_tuple_subgraph_mod

        # Compiled versions of tuple_subgraph_mod and q_tuple_subgraph_mod, created in build_tuple_subgraphs
        self.tuple_subgraph_graphmodule = None
        self.q_tuple_subgraph_graphmodule = None

        self.scale_zp_value_map = {}

        # Initialize the value_map to have dummy values for scales and zero points of 1 and 0, respectively
        # We will never expose values in the graph created with the the dummy values
        for (variable_pairs) in self.output_index_map.keys():
            for (scale_var, zp_var) in variable_pairs:
                self.scale_zp_value_map[scale_var.name_hint] = np.array(1).astype('float32')
                self.scale_zp_value_map[zp_var.name_hint] = np.array(0).astype('int32')

    def build_tuple_subgraphs(self, target, ctx):
        
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]): #TODO: enable AlterOpLayout when fixed
            tuple_subgraph_lib = relay.build(self.tuple_subgraph_mod, target=target)
            q_tuple_subgraph_lib = relay.build(self.tuple_subgraph_mod, target=target)

        self.tuple_subgraph_graphmodule = graph_runtime.GraphModule(tuple_subgraph_lib["default"](ctx))
        self.q_tuple_subgraph_graphmodule = graph_runtime.GraphModule(q_tuple_subgraph_lib["default"](ctx))

    def run_tuple_mod(self, inputs, idx_list):

        # Set the user provided inputs
        for i, inp in enumerate(inputs):
            self.tuple_subgraph_graphmodule.set_input(i, inp)
        
        # Set the scale and zero points
        self.tuple_subgraph_graphmodule.set_input(**self.scale_zp_value_map)
        self.tuple_subgraph_graphmodule.run()

        value_list = []

        for idx in idx_list:
            value_list.append(self.tuple_subgraph_graphmodule.get_output(idx).asnumpy())

        return value_list

    def run_quantized_tuple_mod(self, inputs, current_layer_scale_zps, idx_list):

        # Set user provided inputs
        for i, inp in enumerate(inputs):
            self.q_tuple_subgraph_graphmodule.set_input(i, inp)
        
        # Set the scale and zero points
        self.q_tuple_subgraph_graphmodule.set_input(**self.scale_zp_value_map)
        self.q_tuple_subgraph_graphmodule.set_input(**current_layer_scale_zps)
        self.q_tuple_subgraph_graphmodule.run()

        value_list = []

        for idx in idx_list:
            value_list.append(self.q_tuple_subgraph_graphmodule.get_output(idx).asnumpy())

        return value_list

class Calibrater:
    def __init__(self):
        self.calibration_map = None
        self.quantized_mod = None

        # Tuples containing indices of the inputs and outputs of this layer
        # These will be used in _get_unquantized_layer_inputs, _get_quantized_layer_inputs,
        # _get_unquantized_layer_outputs, and _get_quantized_layer_output
        self.input_tuple_idxs = None
        self.q_input_tuple_idxs = None
        self.output_tuple_idxs = None
        self.q_output_tuple_idxs = None

    def calibrate(self, quantized_mod, calibration_map, target='llvm', ctx=tvm.cpu()):
        """Iterates over every layer in the graph, and sets scale and zero point variables for those layers."""
        self.quantized_mod = quantized_mod
        self.calibration_map = calibration_map
        
        self.calibration_map.build_tuple_subgraphs(target, ctx)
        
        for ((variable_pairs), ((input_tuple_idxs, q_input_tuple_idxs), (output_tuple_idx, q_output_tuple_idx))) in self.calibration_map.output_index_map.items():
            # Save current indices so they can be accessed from helper functions
            self.input_tuple_idxs = input_tuple_idxs
            self.q_input_tuple_idxs = q_input_tuple_idxs
            self.output_tuple_idx = output_tuple_idx
            self.q_input_tuple_idxs = q_output_tuple_idx

            value_dict = self._calibration_callback(variable_pairs)
            # Merge values picked for scale and zp variables to 
            self.calibration_map.scale_zp_value_map.update(value_dict)
        
        # Create the calibrated module using the scales and zero points we just found
        calibrated_func = relay.build_module.bind_params_by_name(quantized_mod['main'], self.calibration_map.scale_zp_value_map)
        calibrated_mod = tvm.ir.IRModule()
        calibrated_mod['main'] = calibrated_func

        optimize = tvm.transform.Sequential(
            [relay.transform.FoldConstant()])
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]): # TODO: enable AlterOpLayout when it's fixed
            calibrated_mod = optimize(calibrated_mod)
        
        return calibrated_mod

    def _calibration_callback(self, variable_pairs):
        """Sets the values of the scale and zero point variables. This function should be implemented in a subclass.
        
        Parameters
        ----------
        variable_pairs : tuple of tuple of relay.var
            Each input to the current quantized layer has a scale and zero point  
        """
        raise NotImplementedError

    def _get_unquantized_layer_inputs(self, inputs):
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
        return self.calibration_map.run_tuple_mod(inputs, self.input_tuple_idxs)

    def _get_quantized_layer_inputs(self, inputs, current_layer_scale_zps):
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
        return self.calibration_map.run_quantized_tuple_mod(inputs, current_layer_scale_zps, self.q_input_tuple_idxs)

    def _get_unquantized_layer_output(self, inputs):
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
        return self.calibration_map.run_tuple_mod(inputs, self.output_tuple_idx)

    def _get_quantized_layer_output(self, inputs, current_layer_scale_zps):
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
        return self.calibration_map.run_tuple_mod(inputs, current_layer_scale_zps, self.q_output_tuple_idxs)
