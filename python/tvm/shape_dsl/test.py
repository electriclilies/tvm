import tvm
from tvm import te
from tvm.runtime.container import ADT
from typing import List

# class ShapeExpr(object):
#     pass

# class ShapeTuple(ShapeExpr):
#     dims : List[int]

#     def __init__(self, dims=[]):
#         if isinstance(dims, int):
#             self.dims = [dims]
#         else:
#             self.dims = dims

# type A = Foo(string, string) | Bar(int) | Null | ...

# enum A {
#     Fo(....)

# }

# type A = {
#     Foo(string, string),
#     bar(...),
#     Null
# }

# Foo(x, y) -> ADT(0, [x, y])
# Bar(w) -> ADT(1, [w])
# Null -> ADT(2, [])

# match a with .... { }

# if a.tag == 0:
#     x = a.fields[0]
#     y = a.fields[1]
#     ...
# else if a.tag == 1:
#     w = a.fields[0]
# else if c.tag == 2:


# (a, b, c) => ADT(0, [a, b, c])

def shape_tuple(*dims, **kwargs):
    new_dims = []
    for i in dims:
        new_dims.append(tvm.tir.IntImm('int32', dims[i]))
    return ADT(0, new_dims)

# my_tuple = ([1, 2, 3, 4])
# print(my_tuple.dims)

def arange_shape_func():
    start = te.var("start")
    stop = te.var("stop")
    step = te.var("step")
    inputs = [start, stop, step]
    out = tvm.tir.Cast("int64", tvm.tir.ceil(tvm.tir.Div(tvm.tir.Sub(start, stop), step)))
    # this causes an error
    func = tvm.tir.PrimFunc(inputs, tvm.tir.stmt.Evaluate(out))
    var = tvm.ir.GlobalVar("my_func")
    mod = tvm.ir.IRModule(functions={ var: func })


@tvm.register_func("transpose_shape")
def handwritten_arange_func(start, stop, step):
    assert ((stop - start) % step) == 0
    rank = int((stop - start) / step)
    return ShapeTuple(rank) # how do I register ShapeTuple with FFI_API and what does register_func actually do

# arange_shape_func()
# sh_tuple = shape_tuple(1, 3, 1)
# print(sh_tuple)
# print(handwritten_arange_func(shape_tuple(1, 3, 1)))

from tvm import relay

x = relay.var('x', dtype="float32", shape=(10, 1))
f = relay.Function([x], x + x + x)
mod = tvm.IRModule.from_expr(f)

fusion_pass = relay.transform.FuseOps()
mod = fusion_pass(mod)
print(mod["main"])
