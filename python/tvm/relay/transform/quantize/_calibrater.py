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
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.transform.quantize import Quantizer

class Calibrater():
    """Calls calibrate_pattern to get scale and zero point values,
    and updates CalibrationInfo."""
    def __init__(self, quantizer, target='llvm', ctx=tvm.cpu(), dataset_manager = None):
        self.quantizer = quantizer

        self.calibration_info = CalibrationInfo(quantizer.tuple_subgraph_func, \
                                                quantizer.q_tuple_subgraph_func, \
                                                quantizer.partition_infos, \
                                                dataset_manager, target, ctx)
    def calibrate(self):
        # Create a map of DFPatternCallback to QuantizerPattern
        pattern_map = {pattern.pattern:pattern for pattern in self.quantizer.patterns}

        for partition_info in self.calibration_info.partition_infos:
            # Set the partition info so we can access it from the callback
            self.calibration_info.set_current_partition_info(partition_info)
            quantizer_pattern = pattern_map[partition_info.pattern]
            
            # Get the values for scales and ZPs in this layer, store
            scale_zps = quantizer_pattern.calibrate_pattern(self.calibration_info)
            self.calibration_info.update_scale_zp_map(scale_zps)
    
        calibrated_func = relay.build_module.bind_params_by_name(self.quantizer.q_tuple_subgraph_func, \
                                                                 self.calibration_info.scale_zp_value_map)
        
        # If num_original_outputs is -1, original output wasn't a tuple
        if (self.quantizer.num_original_outputs == -1):
            calibrated_func = relay.Function(calibrated_func.params, calibrated_func.body.fields[0])
        else:
            new_body = relay.Tuple(calibrated_func.body.fields[0:self.quantizer.num_original_outputs])
            calibrated_func = relay.Function(calibrated_func.params, new_body)

        return calibrated_func

class CalibrationInfo():
    """Helper class that passes information necessary for picking scales and zero points into 
    calibrate_pattern. The state of CalibrationInfo is updated by Calibrater."""
    def __init__(self, tuple_subgraph_func, q_tuple_subgraph_func, partition_infos, \
                 dataset_manager, target, ctx):
        self.tuple_subgraph_func = tuple_subgraph_func
        self.q_tuple_subgraph_func = q_tuple_subgraph_func
        self.partition_infos = partition_infos
        self.dataset_manager = dataset_manager
        self.target = target
        self.ctx = ctx

        self.partition_info = None
        self.input_scale_zps = None
        
        tuple_subgraph_mod = tvm.ir.IRModule.from_expr(self.tuple_subgraph_func)
        q_tuple_subgraph_mod = tvm.ir.IRModule.from_expr(self.q_tuple_subgraph_func)

        self.tuple_subgraph_graphmodule = None
        self.q_tuple_subgraph_graphmodule = None
        self.init_subgraph_graphmodules(tuple_subgraph_mod, q_tuple_subgraph_mod)

        self.scale_zp_value_map = {}
        self.initialize_scale_zp_map()

    def init_subgraph_graphmodules(self, tuple_subgraph_mod, q_tuple_subgraph_mod):
        """Builds the tuple subgraphs so they can be run during calibration."""
        # AlterOpLayout is disabled because it inserts some pads and other ops
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            tuple_subgraph_lib = relay.build(tuple_subgraph_mod, target=self.target)
            q_tuple_subgraph_lib = relay.build(q_tuple_subgraph_mod, target=self.target)

        self.tuple_subgraph_graphmodule = graph_runtime.GraphModule(tuple_subgraph_lib["default"](self.ctx))
        self.q_tuple_subgraph_graphmodule = graph_runtime.GraphModule(q_tuple_subgraph_lib["default"](self.ctx))

    def initialize_scale_zp_map(self):
        """Initializes scales to 1 and zero points to zero. These values will only be used
        to calculate values in the tuple subgraph that are not returned to the user."""
        for p_info in self.partition_infos:
            for count in range(len(p_info.input_scale_zps)):
                scale_var = p_info.input_scale_zps[count][0]
                zp_var = p_info.input_scale_zps[count][1]

                self.scale_zp_value_map[scale_var.name_hint] = np.array(1).astype('float32')
                self.scale_zp_value_map[zp_var.name_hint] = np.array(0).astype('int32')
    
    def set_current_partition_info(self, partition_info):
        """Sets the partition_info for the current iteration, and exposes the
        list of scale and zp variables directly instead of through the partition_info
        object."""
        self.partition_info = partition_info
        self.input_scale_zps = self.partition_info.input_scale_zps

    def update_scale_zp_map(self, new_scale_zps):
        """Updates the Calibrater's scale and zero point map with values returned
        from calibrate_pattern."""
        self.scale_zp_value_map.update(new_scale_zps)

    def _run_tuple_mod(self, inputs, idx_list):
        """Runs the graph that has all the intermediate outputs in it, and extracts
        the values for the current pattern using indices into the tuple."""
        value_list = []

        # Set the user provided inputs
        for i, inp in enumerate(inputs):
            self.tuple_subgraph_graphmodule.set_input(i, inp)

        self.tuple_subgraph_graphmodule.run()

        # Get the correct values out
        for idx in idx_list:
            value_list.append(self.tuple_subgraph_graphmodule.get_output(idx.value).asnumpy())

        return value_list

    def _run_quantized_tuple_mod(self, inputs, current_layer_scale_zps, idx_list):
        """Runs the quantized verion of the graph that has all the intermediate outputs in it,
        and extracts the values for the current pattern using indices into the tuple. Because we
        are running the quantized version, we need to pass in scales and zero points for the current
        pattern."""
        value_list = []

        # Set user provided inputs
        for i, inp in enumerate(inputs):
            self.q_tuple_subgraph_graphmodule.set_input(i, inp)
        
        # Set the scale and zero points
        self.q_tuple_subgraph_graphmodule.set_input(**self.scale_zp_value_map)
        self.q_tuple_subgraph_graphmodule.set_input(**current_layer_scale_zps)

        self.q_tuple_subgraph_graphmodule.run()

        for idx in idx_list:
            value_list.append(self.q_tuple_subgraph_graphmodule.get_output(idx.value).asnumpy())

        return value_list
    
    def get_unquantized_layer_inputs(self, data):
        """Utility function that evaluates the inputs to the current layer and returns the results
        for given inputs. This function should be called from inside calibrate_pattern.

        Parameters
        ----------
        inputs : List
            List of inputs to pass into the mod. Inputs appear in the same order they appeared in the original,
            unquantized function.

        Returns
        -------
        input_values : tuple of numpy.ndarray
            A tuple of the values of inputs to the unquantized layer. If the layer is a binop, there will be two
            elements in the tuple, if an n-op, there will be n elements in the tuple.
        """
        return self._run_tuple_mod(data, self.partition_info.input_idxs)

    def get_quantized_layer_inputs(self, data, current_layer_scale_zps):
        """Utility function that evaluates the quantized inputs to the current quantized layer,
        and returns the results in a tuple. It uses previously set scale and zero points when evaluating
        the graph. This function should be called from inside calibrate_pattern.

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
            A tuple of the values of the inputs to the quantized layer. If the layer is a binop, there will be two
            elements in the tuple, if an n-op, there will be n elements in the tuple.
        """
        return self._run_quantized_tuple_mod(data, current_layer_scale_zps, self.partition_info.input_idxs)

    def get_unquantized_layer_output(self, inputs):
        """Utility function that evaluates the unquantized output of the current layer and returns it.
        This function should be called from inside calibrate_pattern.

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
        This function should be called from inside calibrate_pattern.

        Parameters
        ----------
        inputs : list
            List of inputs to pass into the mod. Inputs appear in the same order they appeared in
            the original, unquantized function.
        
        current_layer_scale_zps: dictionary
            Map from names of scales and zero points you are setting in the current layer to their values.
            This map should be of the same format as the map you return from _calibration_callback.

        Returns
        -------
        output_value : numpy.ndarray
            The output of the quantized layer.
        """
        return self._run_quantized_tuple_mod(data, current_layer_scale_zps, [self.partition_info.output_idx])
