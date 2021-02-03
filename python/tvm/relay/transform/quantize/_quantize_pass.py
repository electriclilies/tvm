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

from typing import List

import tvm
from tvm.relay.transform.quantize import Quantizer, Calibrater, Requantizer, QuantizerPattern
from .. import function_pass

@function_pass(opt_level=5)
class QuantizePass:
    """Explicit pass wrapper around quantization workflow"""
    def __init__(self, quantizer_pattern_list : List[QuantizerPattern], params, \
                 target='llvm', ctx=tvm.cpu(0)):
        self.quantizer_pattern_list = quantizer_pattern_list
        self.params = params
        self.target = target
        self.ctx = ctx

    def transform_function(self, func, _):
        params = {}
        # Extract params that are in this function
        for param in func.params:
            params[param.name_hint] = self.params[param.name_hint]
        quantizer = Quantizer(func, params, self.quantizer_pattern_list)
        calibrater = Calibrater(quantizer, target=self.target, ctx=self.ctx)
        transformed_func = calibrater.calibrate()
        transformed_func = Requantizer().requantize(transformed_func)
        return transformed_func
