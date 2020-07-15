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
import topi.testing
from tvm.relay.testing import run_infer_type

def test_dyn_upsampling():
    n, c, h, w = te.size_var("n"), 16, 32, 32
    scale_h = 2.0
    scale_w = 2.0
    dtype = "float32"
    layout = "NCHW"
    method = "nearest_neighbor"
    align_corners = False
    dshape = (1, ) + (c, w, h)
    #n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    #scale_h = relay.Var("scale_h", relay.TensorType((1, ), "float32"))
    #scale_w = relay.Var("scale_w", relay.TensorType((1, ), "float32"))
    x = relay.Var("x", relay.TensorType(dshape, dtype))
    y = relay.nn.upsampling(x, scale_h=scale_h, scale_w=scale_w, layout=layout, 
                            method=method, align_corners=align_corners)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * 4, dtype)
    func = relay.Function([x], y)
    data = np.random.uniform(size=dshape).astype(dtype)
    ref_res = topi.testing.upsampling_python(data, (scale_h, scale_w), layout)
    verify_func(func, [data], ref_res)

if __name__ == "__main__":
    test_dyn_upsampling()
