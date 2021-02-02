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

from tvm.relay.transform.quantize import CalibrationCallback

import numpy as np

# Calls all the methods of CalibrationCallback to make sure they work OK
class TestCalibrationCallback(CalibrationCallback):
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.scale_value = np.array(2).astype('float32')
        self.zp_value = np.array(0.5).astype('int32')

    def calibrate_pattern(self, calibration_info):
        scale_zp_values = {}
        
        for i in range(len(calibration_info.input_scale_zps)):
            scale_name = calibration_info.input_scale_zps[i][0].name_hint
            scale_zp_values[scale_name] = self.scale_value
            zp_name = calibration_info.input_scale_zps[i][1].name_hint
            scale_zp_values[zp_name] = self.zp_value

        inputs = [np.random.rand(*self.input_shape)]

        calibration_info.get_unquantized_layer_inputs(inputs)
        calibration_info.get_unquantized_layer_output(inputs)
        calibration_info.get_quantized_layer_inputs(inputs, scale_zp_values)
        calibration_info.get_quantized_layer_output(inputs, scale_zp_values)

        return scale_zp_values

def test_calibration_callback():
    raise NotImplementedError # TODO: I'm not sure what the best graph to test this on is. I want to just make sure that everything runs OK


if __name__ == '__main__':
    test_calibration_callback()