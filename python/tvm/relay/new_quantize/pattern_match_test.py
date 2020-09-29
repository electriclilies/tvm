import tvm
from tvm import relay
from tvm.relay import ExprMutator, Call, Var, Constant, TupleGetItem, Function
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.util import get_pad_tuple2d
from tvm.relay.dataflow_pattern import rewrite, wildcard, is_op, DFPatternCallback, is_var, dominates

def get_fn():
    x = relay.var('x', shape=[10])
    y = relay.var('y', shape=[10])
    c = x + y
    c = c + x
    #c = x + y
    #c = c * y
    #c = c + x
    ##c = c / x
    #c = relay.nn.max_pool2d(c)
    #c = c + y
    print(c)
    #is_supported_op1 = is_op("add")(wildcard(), wildcard()) #| is_op("multiply")(wildcard(), wildcard()) | is_op("nn.max_pool2d")(wildcard()) | is_var() # allow constants
    is_supported_op1 = is_op("add")(wildcard(), wildcard()) | is_var()
    #rec_depth = 1000
    #for _ in range(rec_depth): # pull out repeat into a separate func
    #is_supported_op2 = is_op("add")(is_supported_op1, is_supported_op1) | is_op("multiply")(is_supported_op1, is_supported_op1) | is_op("nn.max_pool2d")(is_supported_op1) | is_var()
    #is_supported_op2 = is_op("add")(is_op("add")(wildcard(), wildcard()), is_var())


    # bug :((((
    # this does not work
    #is_supported_op2 = is_op("add")(is_supported_op1, is_supported_op1)
    # this works
    is_supported_op2 = is_op("add")(is_op("add")(wildcard(), wildcard()) | is_var(), is_op("add")(wildcard(), wildcard()) | is_var())
    assert is_supported_op2.match(c)
    
def get_dom():
    #zero_point = relay.const(0, dtype='int32')
    #scale = relay.const(1, dtype='float32')
    #x = relay.var('x')
    #q = relay.qnn.op.dequantize(x, scale, zero_point)
    #q = relay.nn.relu(q)
    #out = relay.qnn.op.quantize(q, scale, zero_point)

    ## Define dom pattern
    #is_dequantize = is_op('qnn.dequantize')(wildcard(), wildcard(), wildcard())
    ##is_unary = is_op('nn.relu')(wildcard()) | is_op('nn.max_pool2d')(wildcard()) | is_op("nn.pad")(wildcard())
    #is_unary_elemwise = (wildcard().has_attr({"TOpPattern": 0}))(wildcard())
    #is_quantize = is_op('qnn.quantize')(wildcard(), wildcard(), wildcard())
    
    #pattern = dominates(is_dequantize, is_unary_elemwise, is_quantize)
    
    #assert pattern.match(out)
    # Pattern
    is_conv2d = is_op('qnn.dequantize')(wildcard(), wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": 0}))(wildcard())
    reduction = is_op('qnn.quantize')(wildcard(), wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Classic Diamond
    inp = relay.var('input')
    weight = relay.var('weight')
    scale = relay.var('scale')
    conv2d = relay.qnn.op.dequantize(inp, weight, scale)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    #out = relu + leaky_relu
    out = relay.qnn.op.quantize(leaky_relu, leaky_relu, leaky_relu)

    assert diamond.match(out)

get_dom()