import tvm
from tvm import te
from typing import List

class ShapeExpr(object):
    pass

class ShapeTuple(ShapeExpr):
    dims : List[int]

    def __init__(self, dims=[]):
        if isinstance(dims, int):
            self.dims = [dims]
        else:
            self.dims = dims

my_tuple = ShapeTuple([1, 2, 3, 4])
print(my_tuple.dims)

def arange_shape_func():
    start = te.var("start")
    stop = te.var("stop")
    step = te.var("step")
    inputs = [start, stop, step]
    out = tvm.tir.Cast("int64", tvm.tir.ceil(tvm.tir.Div(tvm.tir.Sub(start, stop), step)))
    # this causes an error
    func = tvm.tir.PrimFunc(inputs, out)
    mod = tvm.ir.IRModule(functions=[func])
    

@tvm.register_func("transpose_shape")
def handwritten_arange_func(start, stop, step):
    assert ((stop - start) % step) == 0
    rank = int((stop - start) / step)
    return ShapeTuple(rank) # how do I register ShapeTuple with FFI_API and what does register_func actually do

arange_shape_func()
#print(handwritten_arange_func(1, 3, 1).dims)

