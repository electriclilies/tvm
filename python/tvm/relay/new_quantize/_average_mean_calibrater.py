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
from tvm.relay.new_quantize import Calibrater

#import tensorflow.compat.v2 as tf
#import tensorflow_datasets as tfds
#tf.enable_v2_behavior()

import numpy as np

class AverageMeanCalibrater(Calibrater):
    def __init__(self, dataset_manager):
        super().__init__()
        self.dataset_manager = dataset_manager

    def _calibration_callback(self, variable_pairs):
        value_dict = {}
        
        min_sums = np.zeros(shape=(len(variable_pairs)))
        max_sums = np.zeros(shape=(len(variable_pairs)))
        
        while not self.dataset_manager.is_empty():
            # Get the original input from dataset manger, run unquantized graph with those inputs
            image_list, _ = self.dataset_manager.get_next_batch()
            unquantized_inputs = self._get_unquantized_layer_inputs(image_list)
            # Iterate through scale and zp variables 
            for i, unquantized_input in enumerate(unquantized_inputs):            
                # Calculate the average min, max across each batch
                
                min_sums[i] += np.min(unquantized_input)
                max_sums[i] += np.max(unquantized_input)

        self.dataset_manager.reset()

        avg_mins = min_sums / self.dataset_manager.num_batches()
        avg_maxs = max_sums / self.dataset_manager.num_batches()

        # Threshold for quantization of an input to a layer is mean(abs(avg_max), abs(avg_min))
        thresholds = np.mean([np.abs(avg_mins), np.abs(avg_maxs)], axis=0)

        # Since this is a symmetric distribution and we are quantizing to int8, there are 256 bins, and 128 are positive
        scales = thresholds / 128

        for scale_value, (scale_var, zp_var) in zip(scales, variable_pairs):
            value_dict[scale_var.name_hint] = np.array(scale_value).astype('float32')
            value_dict[zp_var.name_hint] = np.array(0).astype('int32')
        
            print("Set ", scale_var.name_hint, " to ", scale_value)
            print("Set ", zp_var.name_hint, " to ", 0)

        return value_dict


# TODO: rename me? also how to deal w layouts. 
class PerChannelAverageMeanCalibrater(Calibrater):
    def __init__(self, dataset_manager):
        super().__init__()
        self.dataset_manager = dataset_manager

    def _calibration_callback(self, variable_pairs):
        value_dict = {}
        
        layer_op = self._get_layer_op()
        print("Layer op is: ", layer_op)
        if (layer_op == relay.op.get('qnn.conv2d')):
            attrs = self._get_layer_attributes()
            channels = attrs['channels']
            
            data_min_sums = 0
            data_max_sums = 0

            # Only the weight scales get per channel quantization
            kernel_min_sums = np.zeros(shape=channels)
            kernel_max_sums = np.zeros(shape=channels)

            while not self.dataset_manager.is_empty():
                # Get the original input from dataset manger, run unquantized graph with those inputs
                image_list, _ = self.dataset_manager.get_next_batch()

                # Unquantized_inputs = [data, weight]
                unquantized_inputs = self._get_unquantized_layer_inputs(image_list)

                data = unquantized_inputs[0]
                weight = unquantized_inputs[1]

                data_min_sums += np.min(data)
                data_max_sums += np.max(data)

                # Get the axes to sum over based on layout
                if attrs['kernel_layout'] == "OIHW":
                    # Want to sum across all axes except 0
                    axes = (1, 2, 3)
                elif attrs['kernel_layout'] == "HWIO":
                    # Want to sum across all axes except 3
                    axes = (0, 1, 2)
                else:
                    raise ValueError("Layout %s not supported in PerChannelAverageMeanCalibrater. ", attrs['weight_layout'])

                # Sum across all axes except channel_axis
                kernel_min_sums += np.max(weight, axis=axes)
                kernel_max_sums += np.max(weight, axis=axes)

            self.dataset_manager.reset()
            
            n_batches = self.dataset_manager.num_batches()
            data_threshold = np.mean([data_min_sums / n_batches, data_max_sums / n_batches], axis=0)
            kernel_threshold = np.mean([kernel_min_sums / n_batches, kernel_max_sums / n_batches], axis=0) # TODO: is this axis right?

            data_scale = data_threshold / 128
            kernel_scale = kernel_threshold / 128
            print("Data shape: ", data.shape)
            print("Weight shape: ", weight.shape)
            print("Kernel Scale shape: ", len(kernel_scale))
            scales = [data_scale, kernel_scale]
            # WHY AM I GETTING TYPE ERRORS??? AUGHHHH
            """
            ((data_scale_var, data_zp_var), (kernel_scale_var, kernel_zp_var)) = variable_pairs
            value_dict[kernel_scale_var.name_hint] = kernel_scale
            value_dict[kernel_]
            """
        else:
            min_sums = np.zeros(shape=(len(variable_pairs)))
            max_sums = np.zeros(shape=(len(variable_pairs)))
            
            while not self.dataset_manager.is_empty():
                # Get the original input from dataset manger, run unquantized graph with those inputs
                image_list, _ = self.dataset_manager.get_next_batch()
                unquantized_inputs = self._get_unquantized_layer_inputs(image_list)
                # Iterate through scale and zp variables 
                for i, unquantized_input in enumerate(unquantized_inputs):            
                    # Calculate the average min, max across each batch
                    # TODO: how to deal with updates and layouts
                    min_sums[i] += np.min(unquantized_input)
                    max_sums[i] += np.max(unquantized_input)

            self.dataset_manager.reset()

            avg_mins = min_sums / self.dataset_manager.num_batches()
            avg_maxs = max_sums / self.dataset_manager.num_batches()

            # Threshold for quantization of an input to a layer is mean(abs(avg_max), abs(avg_min))
            thresholds = np.mean([np.abs(avg_mins), np.abs(avg_maxs)], axis=0)

            # Since this is a symmetric distribution and we are quantizing to int8, there are 256 bins, and 128 are positive
            scales = thresholds / 128

        for scale_value, (scale_var, zp_var) in zip(scales, variable_pairs):
            value_dict[scale_var.name_hint] = np.array(scale_value).astype('float32')
            value_dict[zp_var.name_hint] = np.array(0).astype('int32')
        
            print("Set ", scale_var.name_hint, " to ", scale_value)
            print("Set ", zp_var.name_hint, " to ", 0)

        return value_dict



# TODO: Where to put dataset manager?
class DatasetManager():
    def __init__(self):
        raise NotImplementedError

    def get_next_batch(self):
        raise NotImplementedError

    def num_batches(self):
        raise NotImplementedError

    def is_empty(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class TFDatasetManager(DatasetManager):
    def __init__(self, tf_dataset, batch_size, n_batches):
        self.idx = 0
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.tf_dataset = tf_dataset
        self.tf_iter = iter(self.tf_dataset)
    
    # Returns the next batch of data
    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1
        
        data, label = next(self.tf_iter)
        return [data.numpy()], label.numpy()
    
    def num_batches(self):
        return self.n_batches

    def is_empty(self):
        return self.idx >= self.n_batches

    # TODO: either reset automatically at the end of get batch or have a is_empty util
    def reset(self):
        self.tf_iter = iter(self.tf_dataset)
        self.idx = 0