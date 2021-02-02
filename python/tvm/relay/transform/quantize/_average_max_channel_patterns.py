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

from tvm import relay
from tvm.relay.transform.quantize import Conv2DPattern, Conv2DBiasAddPattern, DensePattern, PerChannelPattern, CalibrationCallback, QuantizerPattern, DatasetManager

import numpy as np

# See AverageMaxCalibrationCallback in python/tvm/relay/new_quantize/_calibration_callback.py for the version that is not per channel
class AverageMaxPerChannelConv2DPattern(Conv2DPattern, PerChannelPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)

    def extract_attrs(self, pre, post, node_map):
        conv2d = node_map[self.conv2d][0]
        weight = node_map[self.conv_weight][0]

        self.get_attrs(conv2d.attrs, weight.checked_type.shape)
        return post
        
    def scale(self, name, is_weight=False):
        if is_weight:
            shape = (self.channels,)
        else:
            shape = ()
        var = relay.var(str(name) + "_scale_" + str(QuantizerPattern.scales_count), shape=shape, dtype='float32')
        QuantizerPattern.scales_count += 1
        return var

    def calibrate_pattern(self, calibration_info):
        self.attr_callback(calibration_info.partition_info.expr)
        
        # Maybe I need the node map?
        scale_zp_values = {}
        
        data_min_sum = 0
        data_max_sum = 0

        weight_min_sums = np.zeros(shape=(self.channels,))
        weight_max_sums = np.zeros(shape=(self.channels,))

        while not calibration_info.dataset_manager.is_empty():
            # Get the original input from dataset manger, run unquantized graph with those inputs
            image_list, _ = calibration_info.dataset_manager.get_next_batch()
            unquantized_inputs = calibration_info.get_unquantized_layer_inputs(image_list)

            data = unquantized_inputs[0]
            weight = unquantized_inputs[1]

            data_min_sum += np.min(data)
            data_max_sum += np.max(data)

            weight_min_sums += np.min(weight, axis=list(range(len(weight.shape))).remove(0))
            weight_max_sums += np.max(weight, axis=list(range(len(weight.shape))).remove(0))

        calibration_info.dataset_manager.reset()
        
        data_min_avg = data_min_sum / calibration_info.dataset_manager.num_batches()
        data_max_avg = data_max_sum / calibration_info.dataset_manager.num_batches()

        weight_min_avgs = weight_min_sums / calibration_info.dataset_manager.num_batches()
        weight_max_avgs = weight_max_sums / calibration_info.dataset_manager.num_batches()
        # Threshold for quantization of an input to a layer is mean(abs(avg_max), abs(avg_min))
        data_threshold = (np.abs(data_min_avg) + np.abs(data_max_avg)) / 2
        weight_thresholds = (np.abs(weight_min_avgs) + np.abs(weight_max_avgs)) / 2

        # Since this is a symmetric distribution and we are quantizing to int8, there are 256 bins, and 128 are positive
        data_scale = data_threshold / 128
        weight_scales = weight_thresholds / 128
        
        data_scale_name = calibration_info.partition_info.input_scale_zps[0][0].name_hint
        data_zp_name = calibration_info.partition_info.input_scale_zps[0][1].name_hint

        scale_zp_values[data_scale_name] = np.array(data_scale).astype('float32')
        scale_zp_values[data_zp_name] = np.array(0).astype('int32')

        weight_scale_name = calibration_info.partition_info.input_scale_zps[1][0].name_hint
        weight_zp_name = calibration_info.partition_info.input_scale_zps[1][1].name_hint

        scale_zp_values[weight_scale_name] = np.array(weight_scales).astype('float32')
        scale_zp_values[weight_zp_name] = np.array(0).astype('int32')

        return scale_zp_values

# TODO: make sure order of imports is correct
class AverageMaxPerChannelConv2DBiasAddPattern(AverageMaxPerChannelConv2DPattern, Conv2DBiasAddPattern):
    def __init__(self, calibration_callback : CalibrationCallback = None):
        super().__init__(calibration_callback)


class AverageMaxPerChannelDensePattern(DensePattern, PerChannelPattern):
    def __init__(self, calibration_callback : CalibrationCallback):
        super().__init__(calibration_callback)

    def extract_attrs(self, pre, post, node_map):
        dense = node_map[self.dense][0]
        weight = node_map[self.weight][0]

        self.get_attrs(dense.attrs, weight.checked_type.shape)
        self.units = self.attrs['units']

        return post
        
    def scale(self, name, is_weight=False):
        if is_weight:
            shape = (self.attrs['units'],)
        else:
            shape = ()
        var = relay.var(str(name) + "_scale_" + str(QuantizerPattern.scales_count), shape=shape, dtype='float32')
        QuantizerPattern.scales_count += 1
        return var

    def calibrate_pattern(self, calibration_info):
        self.attr_callback(calibration_info.partition_info.expr)
        
        # Maybe I need the node map?
        scale_zp_values = {}
        
        data_min_sum = 0
        data_max_sum = 0

        weight_min_sums = np.zeros(shape=(self.attrs['units'],))
        weight_max_sums = np.zeros(shape=(self.attrs['units'],))

        while not calibration_info.dataset_manager.is_empty():
            # Get the original input from dataset manger, run unquantized graph with those inputs
            image_list, _ = calibration_info.dataset_manager.get_next_batch()
            unquantized_inputs = calibration_info.get_unquantized_layer_inputs(image_list)

            data = unquantized_inputs[0]
            weight = unquantized_inputs[1]

            data_min_sum += np.min(data)
            data_max_sum += np.max(data)
            
            weight_min_sums += np.min(weight, axis=1)
            weight_max_sums += np.max(weight, axis=1)

        calibration_info.dataset_manager.reset()
        
        data_min_avg = data_min_sum / calibration_info.dataset_manager.num_batches()
        data_max_avg = data_max_sum / calibration_info.dataset_manager.num_batches()

        weight_min_avgs = weight_min_sums / calibration_info.dataset_manager.num_batches()
        weight_max_avgs = weight_max_sums / calibration_info.dataset_manager.num_batches()

        # Threshold for quantization of an input to a layer is mean(abs(avg_max), abs(avg_min))
        data_threshold = (np.abs(data_min_avg) + np.abs(data_max_avg)) / 2
        weight_thresholds = (np.abs(weight_min_avgs) + np.abs(weight_max_avgs)) / 2

        # Since this is a symmetric distribution and we are quantizing to int8, there are 256 bins, and 128 are positive
        data_scale = data_threshold / 128
        weight_scales = weight_thresholds / 128
        
        data_scale_name = calibration_info.partition_info.input_scale_zps[0][0].name_hint
        data_zp_name = calibration_info.partition_info.input_scale_zps[0][1].name_hint

        scale_zp_values[data_scale_name] = np.array(data_scale).astype('float32')
        scale_zp_values[data_zp_name] = np.array(0).astype('int32')

        weight_scale_name = calibration_info.partition_info.input_scale_zps[1][0].name_hint
        weight_zp_name = calibration_info.partition_info.input_scale_zps[1][1].name_hint

        scale_zp_values[weight_scale_name] = np.array(weight_scales).astype('float32')
        scale_zp_values[weight_zp_name] = np.array(0).astype('int32')

        return scale_zp_values
