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
""" 
Support level10 operator test cases.

"""


import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import ctx_list, run_infer_type
import random
from test_dynamic_op_level3 import verify_func

def test_dyn_broadcast_to():
    dtype = 'uint8'
    rank = 3
    shape_type = 'int64'
    dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
    x_shape = (1,)
    x = relay.Var("x", relay.ty.TensorType(x_shape, dtype))
    z = relay.broadcast_to(x, dyn_shape)
    zz = run_infer_type(z)
    
    assert zz.checked_type == relay.ty.TensorType((relay.Any(),) * rank, dtype)

    func = relay.Function([x, dyn_shape], z)
    
    x = np.random.uniform(size=x_shape).astype(dtype)
    dyn_shape = (1,)*rank
    ref_res = np.broadcast_to(x, dyn_shape)

    verify_func(func, [x, np.array(dyn_shape).astype(shape_type)], ref_res)

if __name__ == "__main__":
    test_dyn_broadcast_to()
