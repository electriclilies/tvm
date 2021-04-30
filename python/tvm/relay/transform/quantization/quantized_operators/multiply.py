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


def generate_quantized_multiply(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    input1_qparams: utils.QParams,
    input2_qparams: utils.QParams,
    simulated: Optional[utils.SimulatedDTypes] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    internal_accumulation_dtype = simulated.value if simulated is not None else accumulation_dtype

    output_qparams = utils.QParams(
        (input1_qparams.scale_factor * input2_qparams.scale_factor),
        relay.const(0, dtype=accumulation_dtype),
        accumulation_dtype,
    )
    input1, input2 = utils.quantize_inputs(
        internal_accumulation_dtype, input1, input1_qparams, input2, input2_qparams
    )
    input1_zero_point, input2_zero_point = utils.cast_all(
        internal_accumulation_dtype, input1_qparams.zero_point, input2_qparams.zero_point
    )
    output_term = (input1 - input1_zero_point) * (input2 - input2_zero_point)

    if dequantize:
        output_term = utils.dequantize_expr(
            internal_accumulation_dtype, output_term, output_qparams
        )

    # TODO: simulate the effects of overflow
    return output_term, output_qparams


def example_multiply_simulated(seed=42):
    np.random.seed(seed=seed)
    a_arr = np.random.uniform(-10, 10, size=(5, 10)).astype("float32")
    b_arr = np.random.uniform(-10, 10, size=(5, 10)).astype("float32")

    var_creator = utils.AffineQuantizationVarCreator()
    a = relay.var("a")
    b = relay.var("b")
    a_qparams = var_creator.get_qparams("a")
    b_qparams = var_creator.get_qparams("b")
    mul_output, output_qparams = generate_quantized_multiply(
        a, b, a_qparams, b_qparams, dequantize=True, simulated=utils.SimulatedDTypes.FLOAT32
    )
    f = relay.Function(
        [
            a,
            b,
            a_qparams.scale_factor,
            a_qparams.zero_point,
            b_qparams.scale_factor,
            b_qparams.zero_point,
        ],
        mul_output,
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
        actual_a_qparams.zero_point,
        actual_b_qparams.scale_factor,
        actual_b_qparams.zero_point,
    ).asnumpy()

    print("Quantized result:")
    print(result)
    print()
    print("FP32 result:")
    print(a_arr * b_arr)


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    example_multiply_simulated(seed=42)
