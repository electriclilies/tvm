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

class CalibrationCallback():
    """Abstract class that defines the API calibration methods."""
    def calibrate_pattern(self, calibration_info):
        raise NotImplementedError

class GlobalCalibrationCallback(CalibrationCallback):
    """Sets the scales and zero points to a user-provided value."""
    def __init__(self, scale_value, zp_value):
        self.scale_value = np.array(scale_value).astype('float32')
        self.zp_value = np.array(zp_value).astype('int32')

    def calibrate_pattern(self, calibration_info):
        scale_zp_values = {}        
        for i in range(len(calibration_info.input_scale_zps)):
            scale_name = calibration_info.input_scale_zps[i][0].name_hint
            scale_zp_values[scale_name] = self.scale_value
            zp_name = calibration_info.input_scale_zps[i][1].name_hint
            scale_zp_values[zp_name] = self.zp_value

        return scale_zp_values

class AverageMaxCalibrationCallback(CalibrationCallback):
    """Calculates scales and zero points by calculating the average of the maxiumum 
    absolute value of the value we are quantizing."""
    def calibrate_pattern(self, calibration_info):
        scale_zp_values = {}
        min_sums = np.zeros(shape=(len(calibration_info.partition_info.input_scale_zps)))
        max_sums = np.zeros(shape=(len(calibration_info.partition_info.input_scale_zps)))

        while not calibration_info.dataset_manager.is_empty():
            # Get the original input from dataset manger, run unquantized graph with those inputs
            image_list, _ = calibration_info.dataset_manager.get_next_batch()
            unquantized_inputs = calibration_info.get_unquantized_layer_inputs(image_list)

            # Iterate through scale and zp variables 
            for i, unquantized_input in enumerate(unquantized_inputs):
                # Calculate the average min, max across each batch
                min_sums[i] += np.min(unquantized_input)
                max_sums[i] += np.max(unquantized_input)

        calibration_info.dataset_manager.reset()

        avg_mins = min_sums / calibration_info.dataset_manager.num_batches()
        avg_maxs = max_sums / calibration_info.dataset_manager.num_batches()

        # Threshold for quantization of an input to a layer is mean(abs(avg_max), abs(avg_min))
        thresholds = np.mean([np.abs(avg_mins), np.abs(avg_maxs)], axis=0)

        # Since this is a symmetric distribution and we are quantizing to int8, 
        # there are 256 bins, and 128 are positive
        scales = thresholds / 128

        for i, scale_value in enumerate(scales):
            scale_name = calibration_info.partition_info.input_scale_zps[i][0].name_hint
            scale_zp_values[scale_name] = np.array(scale_value).astype('float32')
            zp_name = calibration_info.partition_info.input_scale_zps[i][1].name_hint
            scale_zp_values[zp_name] = np.array(0).astype('int32')

            print("Set ", scale_name, " to ", scale_value)
            print("Set ", zp_name, " to ", 0)

        return scale_zp_values
