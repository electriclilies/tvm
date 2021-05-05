from typing import *

import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    _DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from tvm.relay.op import nn, tensor
from tvm.relay.qnn.python_operators import utils


def generate_generic_quantized_add_or_subtract(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.AffineQParams],
    simulated: Optional[utils.SimulatedDTypes] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
    mode: str = "add",
) -> Tuple[tvm.relay.Expr, utils.AffineQParams]:
    internal_accumulation_dtype = simulated.value if simulated is not None else accumulation_dtype

    input1, input2 = utils.quantize_inputs(
        internal_accumulation_dtype, input1, output_qparams, input2, output_qparams
    )

    if mode == "add":
        output_term = (
            input1 + input2 - utils.cast_all(internal_accumulation_dtype, output_qparams.zero_point)
        )
    elif mode == "sub":
        output_term = (
            input1 - input2 + utils.cast_all(internal_accumulation_dtype, output_qparams.zero_point)
        )
    else:
        raise ValueError("Only support addition and subtraction")

    if dequantize:
        output_term = utils.dequantize_expr(
            internal_accumulation_dtype, output_term, output_qparams
        )

    # TODO: simulate the effects of overflow
    return output_term, output_qparams


def generate_quantized_add(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.AffineQParams],
    simulated: Optional[utils.SimulatedDTypes] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.AffineQParams]:
    return generate_generic_quantized_add_or_subtract(
        input1=input1,
        input2=input2,
        output_qparams=output_qparams,
        simulated=simulated,
        accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
        mode="add",
    )


def generate_quantized_sub(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.AffineQParams],
    simulated: Optional[utils.SimulatedDTypes] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.AffineQParams]:
    return generate_generic_quantized_add_or_subtract(
        input1=input1,
        input2=input2,
        output_qparams=output_qparams,
        simulated=simulated,
        accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
        mode="sub",
    )


def example_add_simulated(seed=42):
    np.random.seed(seed=seed)
    a_arr = np.random.uniform(-10, 10, size=(5, 10)).astype("float32")
    b_arr = np.random.uniform(-10, 10, size=(5, 10)).astype("float32")

    var_creator = utils.AffineQuantizationVarCreator()
    a = relay.var("a")
    b = relay.var("b")
    output_qparams = var_creator.get_qparams("output")

    add_output, output_qparams = generate_quantized_add(
        a, b, output_qparams, simulated=utils.SimulatedDTypes.FLOAT32
    )
    f = relay.Function(
        [a, b, output_qparams.scale_factor, output_qparams.zero_point],
        add_output,
    )
    print(f)

    actual_output_qparams = utils.get_quantization_parameters(a_arr + b_arr, True, 8)

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        a_arr, b_arr, actual_output_qparams.scale_factor, actual_output_qparams.zero_point
    ).asnumpy()

    print("Quantized result:")
    print(result)
    print()
    print("FP32 result:")
    print(a_arr + b_arr)


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    example_add_simulated(seed=42)
