from typing import *

import numpy as np
import torch
import tvm
from torch import nn
from tvm import relay
from tvm.relay.qnn.python_operators import utils


def get_axis_from_layout(dimension_name: str, layout: str):
    if sorted(layout) != sorted("NCHW") or sorted(layout) != sorted("OIHW"):
        raise ValueError(f"Unknown layout {data_layout} need permutation of NCHW or OIHW")

    try:
        return layout.index(dimension_name)
    except ValueError:
        raise ValueError(f"Unknown dimension {dimension_name} for layout {layout}")


def generate_generic_quantized_conv2d(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: utils.QParams,
    weight_qparams: utils.QParams,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    strides: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    kernel_size: Optional[Tuple[int, int]] = None,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: str = "",
    simulated: Optional[utils.SimulatedDTypes] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
    bias: Optional[tvm.relay.Expr] = None,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    if in_channels is None:
        in_channels_axis = get_axis_from_layout("C", data_layout)
        in_channels = weight.checked_type.shape[in_channels_axis]
    if out_channels is None:
        out_channels_axis = get_axis_from_layout("O", kernel_layout)
        out_channels = weight.checked_type.shape[out_channels_axis]
    if kernel_size is None:
        kernel_height_axis = get_axis_from_layoutH("H", kernel_layout)
        kernel_width_axis = get_axis_from_layout("W", kernel_layout)
        kernel_size = (
            weight.checked_type.shape[kernel_height_axis],
            weight.checked_type.shape[kernel_width_axis],
        )

    internal_accumulation_dtype = simulated.value if simulated is not None else accumulation_dtype

    data, weight = utils.quantize_inputs(
        internal_accumulation_dtype, data, data_qparams, weight, weight_qparams
    )

    pad_n = (0, 0)
    pad_c = (0, 0)
    pad_h = (padding[0], padding[0])
    pad_w = (padding[1], padding[1])

    if sorted(data_layout) != sorted("NCHW"):
        raise ValueError(f"Unknown layout {data_layout} need dimension: N, H, W, C")

    padding = []
    for dimension_name in data_layout:
        if dimension_name == "N":
            padding.append(pad_n)
        elif dimension_name == "C":
            padding.append(pad_c)
        elif dimension_name == "H":
            padding.append(pad_h)
        else:
            padding.append(pad_w)

    padded_data = relay.nn.pad(
        data,
        tuple(padding),
        pad_value=data_qparams.zero_point,
    )

    first_term = relay.nn.conv2d(
        padded_data,
        weight,
        strides=strides,
        padding=(0, 0),
        dilation=dilation,
        groups=groups,
        channels=out_channels,
        kernel_size=kernel_size,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=out_layout,
        out_dtype=internal_accumulation_dtype,
    )

    # TODO: consider sum/avg pooling and then reduction along channels instead of this
    second_term = (
        relay.nn.conv2d(
            padded_data,
            relay.ones_like(weight),
            strides=strides,
            padding=(0, 0),
            dilation=dilation,
            groups=groups,
            channels=out_channels,
            kernel_size=kernel_size,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_layout=out_layout,
            out_dtype=internal_accumulation_dtype,
        )
        * utils.cast_all(internal_accumulation_dtype, weight_qparams.zero_point)
    )

    if kernel_layout == "OIHW":
        third_term = relay.sum(weight, axis=0, keepdims=False, exclude=True)
    else:
        third_term = relay.sum(weight, axis=1, keepdims=False, exclude=True)
    third_term *= utils.cast_all(internal_accumulation_dtype, data_qparams.zero_point)

    if data_layout == "NCHW":
        third_term = relay.reshape(third_term, (1, out_channels, 1, 1))
    else:
        third_term = relay.reshape(third_term, (1, 1, 1, out_channels))

    data, weight, data_zero_point, weight_zero_point = utils.cast_all(
        internal_accumulation_dtype,
        data,
        weight,
        data_qparams.zero_point,
        weight_qparams.zero_point,
    )

    fourth_term = (
        data_zero_point
        * weight_zero_point
        * relay.const(kernel_size[0], dtype=internal_accumulation_dtype)
        * relay.const(kernel_size[1], dtype=internal_accumulation_dtype)
        * relay.const(in_channels, dtype=internal_accumulation_dtype)
        / relay.const(groups, dtype=internal_accumulation_dtype)
    )

    output_qparams = utils.QParams(
        data_qparams.scale_factor * weight_qparams.scale_factor,
        relay.const(0, dtype=accumulation_dtype),
        accumulation_dtype,
    )

    # Don't do operations sequentially to better use parallelism
    output_term = (first_term - second_term) - (third_term - fourth_term)

    if bias is not None:
        bias = utils.quantize_inputs(internal_accumulation_dtype, bias, output_qparams)
        output_term += bias

    if dequantize:
        output_term = utils.dequantize_expr(
            internal_accumulation_dtype, output_term, output_qparams
        )

    return output_term, output_qparams


def example_conv_no_zp(in_channels, out_channels, img_height, img_width, groups=2, seed=42):
    np.random.seed(seed=seed)
    kernel_size = 1
    padding = 1

    # NCHW tensors, OIHW kernel
    data_arr = np.random.uniform(-5, 10, size=(1, in_channels, img_height, img_width)).astype(
        "float32"
    )
    weight_arr = np.random.uniform(
        -1, 2, size=(out_channels, in_channels // groups, kernel_size, kernel_size)
    ).astype("float32")

    # bias_arr = np.random.uniform(-100, 100, size=(n, out_units)).astype("float32")

    var_creator = utils.AffineQuantizationVarCreator()
    data = relay.var("data")
    weight = relay.var("weight")
    data_qparams = var_creator.get_qparams("conv_data")
    weight_qparams = var_creator.get_qparams("conv_weight")
    output_tensor, output_qparams = generate_generic_quantized_conv2d(
        data,
        weight,
        data_qparams,
        weight_qparams,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        out_channels=out_channels,
        in_channels=in_channels,
        groups=groups,
        simulated=utils.SimulatedDTypes.FLOAT32,
        dequantize=True,
    )

    f = relay.Function(
        [
            data,
            weight,
            data_qparams.scale_factor,
            data_qparams.zero_point,
            weight_qparams.scale_factor,
            weight_qparams.zero_point,
        ],
        output_tensor,
    )
    print(f)

    actual_data_qparams = utils.get_quantization_parameters(data_arr, True, 8)
    actual_weight_qparams = utils.get_quantization_parameters(weight_arr, True, 8)

    print(actual_data_qparams)
    print(actual_weight_qparams)

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        data_arr,
        weight_arr,
        actual_data_qparams.scale_factor,
        actual_data_qparams.zero_point,
        actual_weight_qparams.scale_factor,
        actual_weight_qparams.zero_point,
    ).asnumpy()

    print("Quantized result:")
    q_result = torch.Tensor(result)
    print(q_result)
    print()

    print("FP32 result:")
    fp32_result = nn.functional.conv2d(
        torch.Tensor(data_arr), torch.Tensor(weight_arr), padding=(padding, padding), groups=groups
    )
    print(fp32_result)
    print()
    print("Difference:")
    print(q_result - fp32_result)


if __name__ == "__main__":
    example_conv_no_zp(4, 2, 5, 5, groups=2)
