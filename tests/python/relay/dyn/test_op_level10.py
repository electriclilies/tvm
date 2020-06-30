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
    dyn_shape = np.array((1,)*rank)
    ref_res = np.broadcast_to(x, dyn_shape)
    for target, ctx in ctx_list():
        for kind in ["vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x,dyn_shape)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


def test_dyn_one_hot():
    def _get_oshape(indices_shape, depth, axis):
        oshape = []
        true_axis = len(indices_shape) if axis == -1 else axis
        ndim = len(indices_shape) + 1
        indices_index = 0
        for i in range(0, ndim):
            if i == true_axis:
                oshape.append(depth)
            else:
                oshape.append(indices_shape[indices_index])
                indices_index += 1

        return oshape

    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        on_value_const = relay.const(on_value)
        off_value_const = relay.const(off_value)
        depth_var = relay.var("depth", relay.TensorType((1,), "int32"))
        out = relay.dyn.one_hot(indices, on_value_const, off_value_const, depth_var, axis, dtype)
        checked = run_infer_type(out)
        
        print(checked.checked_type)
        #assert checked.checked_type == relay.ty.TensorType(_get_oshape(indices_shape, depth, axis), dtype)
        func = relay.Function([indices, depth_var], out)
        indices_np = np.random.randint(0, depth, size=indices_shape).astype("int32")
        out_np = topi.testing.one_hot(indices_np, on_value, off_value, depth, axis, dtype)

        for target, ctx in ctx_list():
            for kind in ["vm", "debug"]:
                mod = tvn.ir.IRModule.from_expr(func)
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                out_relay = intrp.evaluate(func)(indices_np, depth)
                tvm.testing.assert_allclose(out_relay.asnumpy(), out_np)

    _verify((3,), 3, 1, 0, -1, "int32")
    _verify((3,), 3, 1.0, 0.0, -1, "float32")
    _verify((2, 2), 5, 2, -2, 0, "int32")
    _verify((2, 2), 5, 0.5, -0.5, 1, "float32")
    _verify((3, 2, 4, 5), 6, 1, 0, 1, "int32")
    _verify((3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")


if __name__ == "__main__":
    test_dyn_broadcast_to()
    test_dyn_one_hot()