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
from tvm import te
import topi.testing
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import ctx_list, run_infer_type
import topi
import topi.testing
import random

def test_dyn_broadcast_to():
    dtype = 'uint8'
    rank = 3
    dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), 'int64'))
    x_shape = (1,)
    x = relay.Var("x", relay.ty.TensorType(x_shape, dtype))
    z = relay.dyn.broadcast_to(x, shape=dyn_shape)
    zz = run_infer_type(z)
    
    assert zz.checked_type == relay.ty.TensorType((relay.Any(),) * rank, dtype)

    func = relay.Function([x, dyn_shape], z)
    
    x = np.random.uniform(size=x_shape).astype(dtype)
    dyn_shape = (1,)*rank
    print(x)
    print(dyn_shape)
    ref_res = np.broadcast_to(x, dyn_shape)
    for target, ctx in ctx_list():
        for kind in ["vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x,dyn_shape)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

def test_broadcast_to():
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape , dtype))
    z = relay.broadcast_to(x, shape=shape_like)
    zz = run_infer_type(z)
    print(zz.checked_type)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x], z)
    x = np.random.uniform(size=shape).astype(dtype)
    ref_res = np.broadcast_to(x, shape_like)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

#print("TEST BROADCAST_TO")
#test_broadcast_to()
print("TEST DYNAMIC BROADCAST_TO")
test_dyn_broadcast_to()
