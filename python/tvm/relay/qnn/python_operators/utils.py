from enum import Enum, EnumMeta
from typing import *

import numpy as np
import tvm
from tvm import relay


class ContainsEnum(EnumMeta):
    """Base class enum which allows using the python keyword 'in' for element access.

    E.g. 'float32' in SimulatedDTypes is now True.
    """

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class SimulatedDTypes(Enum, metaclass=ContainsEnum):
    """Represents data types for simulated quantization.

    Simulated quantization si where non-integral types hold quantized values for
    type interoperability.
    """

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT = FLOAT32
    DOUBLE = FLOAT64


AffineQParams = NamedTuple(
    "AffineQParams",
    [("scale_factor", tvm.relay.Expr), ("zero_point", tvm.relay.Expr), ("dtype", str)],
)


class AffineQuantizationVarCreator:
    """Class which manages references to our AffineQParams and can insert state."""

    def __init__(self):
        self.ref_count = 0
        self.AffineQParams = []

    def get_qparams(self, name_hint: str, dtype: str = "int8") -> AffineQParams:
        scale = relay.var(f"{name_hint}.scale", type_annotation="float32")
        zero_point = relay.var(f"{name_hint}.zero_point", type_annotation="int8")
        qparam = AffineQParams(scale, zero_point, dtype)
        self.AffineQParams.append(qparam)
        self.ref_count += 1
        return qparam


def cast_all(dtype: str, *args: List[relay.Expr]) -> Union[List[relay.Expr], relay.Expr]:
    """Utility function for casting a lot of things in relay. Handles unpacking well."""
    result = [relay.cast(data, dtype) for data in args]
    if len(result) == 1:
        result = result[0]
    return result


def quantize_inputs(
    simulated: bool,
    internal_accumulation_dtype: str,
    *args: Union[relay.Expr, AffineQParams],
) -> Union[List[relay.Expr], relay.Expr]:
    """Runs given relay nodes through appropriate quantization nodes and returns the relevant relay exp.

    Handles Python unpacking well. E.g.

    a_quantize, b_quantize, c_quantize = quantize_inputs('float32', a, a_qparams, b, b_qparams, c, c_qparams).
    """

    if len(args) % 2 != 0:
        raise ValueError(
            f"Expected alternating expressions and AffineQParams but have even number of entries: {len(args)}"
        )

    # This means use simulated operations
    if simulated:
        quantize_op = relay.qnn.op.simulated_quantize
    else:
        quantize_op = relay.qnn.op.quantize

    quantized_expr = []
    for i in range(len(args) // 2):
        cur_expr = args[2 * i]
        cur_qparam = args[2 * i + 1]

        if not isinstance(cur_expr, relay.Expr):
            raise ValueError(
                f"Expected alternating relay.Expr and AffineQParams! Got a {type(cur_expr)} when expecting relay.Expr"
            )
        if not isinstance(cur_qparam, AffineQParams):
            raise ValueError(
                f"Expected alternating relay.Expr and AffineQParams! Got a {type(cur_expr)} when expecting AffineQParams"
            )

        quantized_value = quantize_op(
            data=cur_expr,
            output_scale=cur_qparam.scale_factor,
            output_zero_point=cur_qparam.zero_point,
            out_dtype=cur_qparam.dtype,
        )

        if not simulated:
            quantized_value = relay.op.cast(quantized_value, internal_accumulation_dtype)
        quantized_expr.append(quantized_value)

    if len(quantized_expr) == 1:
        quantized_expr = quantized_expr[0]
    return quantized_expr


def dequantize_expr(
    simulated: bool,
    expr: relay.Expr,
    qparam: AffineQParams,
) -> Tuple[relay.Expr]:
    """Dequantize the given relay node."""
    if simulated:
        return relay.qnn.op.simulated_dequantize(
            data=expr,
            input_scale=qparam.scale_factor,
            input_zero_point=qparam.zero_point,
            in_dtype=qparam.dtype,
        )
    else:
        return relay.qnn.op.dequantize(
            data=expr,
            input_scale=qparam.scale_factor,
            input_zero_point=qparam.zero_point,
            in_dtype=qparam.dtype,
        )


def get_quantization_parameters(
    data: np.array,
    signed: bool,
    nbits: int,
    per_channel: Optional[int] = None,
    symmetric: bool = False,
    as_relay: bool = True,
) -> Union[AffineQParams, Tuple]:
    """Given a numpy array calculates the optimal AffineQParams."""

    if per_channel is not None:
        raise ValueError("We do not support per channel quantization right now!")

    if not signed or nbits != 8:
        raise ValueError("Only support int8 right now.")

    data = data.flatten()
    rmin, rmax = data.min(), data.max()

    # Always need to represent 0!
    if rmax > 0:
        rmin = min(rmin, 0)
    else:
        rmax = max(rmax, 0)

    scale = (rmax - rmin) / (2 ** nbits - 1)
    if symmetric:
        zero_point = 2 ** (nbits - 1)
    else:
        # This zero point will be based on an unsigned version of nbits, we'll adjust later
        zero_point = abs(rmin // scale)

        # Readjust scale to max sure we can represent both the min and max still
        rmax_scale = rmax / (2 ** nbits - 1 - zero_point)
        rmin_scale = abs(rmin / zero_point)
        scale = max(rmax_scale, rmin_scale)
    if signed:
        zero_point -= 2 ** (nbits - 1)

    if as_relay:
        return AffineQParams(
            relay.const(np.float32(scale)),
            relay.const(np.int8(zero_point)),
            "int8",
        )
    else:
        return (
            np.float32(scale),
            np.int8(zero_point),
            "int8",
        )


def numpy_quantize_values(
    arr: np.array,
    signed: bool,
    nbits: int,
    per_channel: Optional[int] = None,
    symmetric: bool = False,
) -> Tuple[np.array, AffineQParams]:
    """Quantize the given input numpy array and return the quantized array and qparams."""
    a = relay.var("a")
    AffineQParams = get_quantization_parameters(arr, signed, nbits, per_channel, symmetric)
    out = quantize_inputs("float32", a, AffineQParams)

    f = relay.Function([a], out)
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    return intrp.evaluate(f)(arr).asnumpy(), AffineQParams


def numpy_dequantize_values(arr: np.array, qparam: AffineQParams):
    """Dequantize the given numpy array given AffineQParams."""
    a = relay.var("a")
    out = dequantize_expr("float32", a, qparam)

    f = relay.Function([a], out)
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    return intrp.evaluate(f)(arr).asnumpy()
