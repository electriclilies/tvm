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
# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The namespace containing quantization and calibration passes"""
from ._dataset_manager import DatasetManager, TFDatasetManager
from ._calibration_callback import CalibrationCallback, GlobalCalibrationCallback, \
     AverageMaxCalibrationCallback
from ._quantizer_patterns import QuantizerPattern, Conv2DBiasAddPattern, Conv2DPattern, \
     DensePattern, AddPattern, MultiplyPattern, PerChannelPattern, partition_outputs, \
     skip_partitions, rewrite_partitions, lower_partitions
from ._average_max_channel_patterns import AverageMaxPerChannelConv2DBiasAddPattern, \
     AverageMaxPerChannelConv2DPattern, AverageMaxPerChannelDensePattern
from ._quantizer import Quantizer
from ._calibrater import Calibrater
from ._requantizer import Requantizer

from . import _ffi as ffi
