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
""" Support level2 dynamic operator test cases.
"""

import numpy as np
import tvm
from tvm import relay
from tvm import te
from tvm.relay.testing import ctx_list
import random
from test_dynamic_op_level3 import verify_func

def test_dyn_upsampling():
    n, c, h, w = te.size_var("n"), 16, 32, 32
    scale_h = 2.0
    scale_w = 2.0
    dtype = "float32"
    layout = "NCWH"
    method = "nearest_neighbor"
    align_corners = False

    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")

    x = relay.Var("x", relay.TensorType((n, c, h, w), dtype))
    y = relay.nn.upsampling(x, scale_h=scale_h, scale_w=scale_w, layout=layout, 
                            method=method, align_corners=align_corners)
    #print(y)
    #yy = run_infer_type(y)
    #assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * 4, dtype)

if __name__ == "__main__":
    test_dyn_upsampling()