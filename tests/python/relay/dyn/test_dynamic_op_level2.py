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
import tvm.topi.testing
from tvm.relay.testing import run_infer_type

def test_dyn_upsampling():
    def verify_upsampling(dshape, scale_h, scale_w, layout, method, align_corners=False):
        x_data = np.random.uniform(size=dshape).astype("float32")

        if layout == "NCHW":
            (n, c, h, w) = dshape
        elif layout == "NHWC":
            (n, h, w, c) = dshape
        
        if method == "nearest_neighbor":
            ref_res = tvm.topi.testing.upsampling_python(x_data, (scale_h, scale_w), layout)
        else:
            ref_res = tvm.topi.testing.bilinear_resize_python(x_data, (int(round(h*scale_h)),
                                                  int(round(w*scale_w))), layout)
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        scale_h_var = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w_var = relay.var("scale_h", relay.TensorType((), "float32"))

        z = relay.nn.upsampling(x, scale_h_var, scale_w_var, method=method, layout=layout, align_corners=align_corners)
        zz = run_infer_type(z)
        func = relay.Function([x, scale_h_var, scale_w_var], z)

        for target, ctx in ctx_list():
             if "llvm" not in target: continue
             for kind in ["vm", "debug"]:
                 mod = tvm.ir.IRModule.from_expr(func)
                 intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                 op_res = intrp.evaluate()(x_data, np.array(scale_h).astype("float32"), np.array(scale_w).astype("float32"))
                 tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-4, atol=1e-6)

    verify_upsampling((1, 16, 32, 32), 2.0, 2.0,"NCHW", "nearest_neighbor")
    print("one done")
    verify_upsampling((1, 16, 32, 32), 2.0, 2.0,"NCHW", "bilinear", True)
    print("two done")
    verify_upsampling((1, 16, 32, 32), 2.0, 2.0, "NHWC", "nearest_neighbor")
    print("three done")
    verify_upsampling((1, 16, 32, 32), 2.0, 2.0,"NHWC", "bilinear", True)

    print("exact tests from static ops pass")
    for method in ["bilinear", "nearest_neighbor"]:
        for layout in ["NCHW", "NHWC"]:
            print("method: ", method)
            print("layout: ", layout)
            verify_upsampling((1, 16, 32, 32), 2.0, 2.0, layout, method)



# tests upsampling type inference with scale_h and scale_w as variables
def test_upsampling_infer_type():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")

    data = relay.var("data", relay.TensorType((n, c, h, w), "int8"))
    scale_h = relay.Var("scale_h", relay.TensorType((), "float32"))
    scale_w = relay.Var("scale_w", relay.TensorType((), "float32"))
    
    z = relay.nn.upsampling(data, scale_h, scale_w)
    zz = run_infer_type(z)
    print(zz.checked_type)
    assert zz.checked_type == relay.TensorType((n, c, relay.Any(), relay.Any()), "int8")
    
    # Doesn't work right now because of BijectiveShape being terrible
    """
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    data = relay.var("data", relay.TensorType((n, h, w, c), "int8"))
    z2 = relay.nn.upsampling(data, scale_h, scale_w, layout="NHWC")
    zz2 = run_infer_type(z2)
    assert zz2.checked_type == relay.TensorType((n, relay.Any(), relay.Any(), c), "int8")
    """

#tests upsampling type inference with scale_h passed in as a constant and scale_w as a variable
def test_upsampling_infer_type_const():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")

    data = relay.var("data", relay.TensorType((n, c, h, w), "int8"))
    scale_h = relay.Var("scale_h", relay.TensorType((), "float32"))
    scale_w = relay.Var("scale_w", relay.TensorType((), "float32"))

    z = relay.nn.upsampling(data, 2.0, scale_w)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, relay.Any(), relay.Any()), "int8")

def test_dyn_pad():
    def verify_pad(dshape, pad_width, dtype):
        
        x = relay.var("x", relay.TensorType(dshape, dtype))
        pad_width_var = relay.var("pad_width_var", relay.TensorType((len(dshape), 2), 'int64'))
        
        y = relay.nn.pad(x, pad_width_var)
        yy = run_infer_type(y)

        assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * len(dshape), dtype)
       
        func = relay.Function([x, pad_width_var], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = np.pad(data, pad_width, 'constant')
        verify_func(func, [data, pad_width], ref_res)
    
    verify_pad((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), "int32")
    #verify_pad((4, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), "int32")
if __name__ == "__main__":
    test_upsampling_infer_type()
    test_upsampling_infer_type_const()
    test_dyn_upsampling()
    #test_dyn_pad()
