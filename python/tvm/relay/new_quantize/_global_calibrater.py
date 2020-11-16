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
from tvm.relay.new_quantize import Calibrater

import numpy as np

class GlobalCalibrater(Calibrater):

    def __init__(self, scale_value, zp_value):
        super().__init__()
        self.scale_value = np.array(scale_value).astype('float32')
        self.zp_value = np.array(zp_value).astype('int32')
    
    def _calibration_callback(self, variable_pairs):
        value_dict = {}
        for (scale_var, zp_var) in variable_pairs:
            value_dict[scale_var.name_hint] = self.scale_value
            value_dict[zp_var.name_hint] = self.zp_value

        return value_dict
