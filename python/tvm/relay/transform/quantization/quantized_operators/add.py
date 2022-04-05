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
from tvm.relay.transform.quantization.quantized_operators import utils


def generate_generic_quantized_add_or_subtract(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    internal_accumulation_dtype: str = "float32",
    simulated_accumulation_dtype: str = "int32",
    dequantize: bool = True,
    mode: str = "add",
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    if output_qparams is None and (input1_qparams is None or input2_qparams is None):
        raise ValueError(
            "Must give either the output qparams or both input qparams to infer output qparams!"
        )

    if output_qparams is None:
        output_qparams = utils.QParams(
            (input1_qparams.scale_factor + input2_qparams.scale_factor),
            relay.const(0, dtype=simulated_accumulation_dtype),
            simulated_accumulation_dtype,
        )
    input1, input2 = utils.quantize_inputs(
        internal_accumulation_dtype, input1, output_qparams, input2, output_qparams
    )

    input1, input2 = utils.cast_all(internal_accumulation_dtype, input1, input2)

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


def generate_static_quantized_add(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_add(
        input1,
        input2,
        output_qparams,
        input1_qparams=input1_qparams,
        input2_qparams=input2_qparams,
        internal_accumulation_dtype=accumulation_dtype,
        simulated_accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
        mode="add",
    )


def generate_simulated_quantized_add(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_add(
        input1,
        input2,
        output_qparams,
        input1_qparams=input1_qparams,
        input2_qparams=input2_qparams,
        internal_accumulation_dtype="float32",
        simulated_accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
        mode="add",
    )


def generate_static_quantized_sub(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_add(
        input1,
        input2,
        output_qparams,
        input1_qparams=input1_qparams,
        input2_qparams=input2_qparams,
        internal_accumulation_dtype=accumulation_dtype,
        simulated_accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
        mode="sub",
    )


def generate_simulated_quantized_sub(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_add(
        input1,
        input2,
        output_qparams,
        input1_qparams=input1_qparams,
        input2_qparams=input2_qparams,
        internal_accumulation_dtype="float32",
        simulated_accumulation_dtype=accumulation_dtype,
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
    a_qparams = var_creator.get_qparams("a")
    b_qparams = var_creator.get_qparams("b")
    add_output, output_qparams = generate_simulated_quantized_add(
        a, b, None, a_qparams, b_qparams, dequantize=True
    )
    f = relay.Function(
        [
            a,
            b,
            a_qparams.scale_factor,
            b_qparams.scale_factor,
        ],
        add_output,
    )
    print(f)

    actual_a_qparams = utils.get_quantization_parameters(a_arr, True, 8)
    actual_b_qparams = utils.get_quantization_parameters(b_arr, True, 8)

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        a_arr,
        b_arr,
        actual_a_qparams.scale_factor,
        actual_b_qparams.scale_factor,
    ).asnumpy()

    print("Quantized result:")
    print(result)
    print()
    print("FP32 result:")
    print(a_arr + b_arr)


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    example_add_simulated(seed=42)
