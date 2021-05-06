from collections import namedtuple

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.python_operators import add, utils

TestCase = namedtuple("TestCase", ["shape", "bits", "symmetric"])
TestCase.__test__ = False


def relay_quantized_add_or_sub(
    a_data: np.array,
    b_data: np.array,
    simulated_dtype: utils.SimulatedDTypes = utils.SimulatedDTypes.FLOAT32,
    bits: int = 8,
    signed: bool = True,
    symmetric_output=True,
    use_add: bool = True,
):
    actual_output_qparams = utils.get_quantization_parameters(
        a_data + b_data, True, bits, symmetric=symmetric_output
    )

    var_creator = utils.AffineQuantizationVarCreator()
    a = relay.var("a", shape=a_data.shape, dtype="float32")
    b = relay.var("b", shape=b_data.shape, dtype="float32")
    output_qparams = var_creator.get_qparams("output")

    if use_add:
        output, output_qparams = add.generate_quantized_add(
            a, b, output_qparams, simulated_dtype=simulated_dtype, dequantize=True
        )
    else:
        output, output_qparams = add.generate_quantized_sub(
            a, b, output_qparams, simulated_dtype=simulated_dtype, dequantize=True
        )

    f = relay.Function(
        [a, b, output_qparams.scale_factor, output_qparams.zero_point],
        output,
    )

    mod = tvm.ir.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    f = mod["main"]
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        a_data, b_data, actual_output_qparams.scale_factor, actual_output_qparams.zero_point
    ).asnumpy()

    return result


def run_test_case(test_case: TestCase, use_add: bool, simulated: bool):
    shape, bits, symmetric = test_case

    a = np.random.uniform(-5, 10, size=shape).astype("float32")
    b = np.random.uniform(-5, 3, size=shape).astype("float32")

    if use_add:
        expected_output = a + b
    else:
        expected_output = a - b

    simulated_dtype = utils.SimulatedDTypes.FLOAT32 if simulated else None
    actual_output = relay_quantized_add_or_sub(
        a,
        b,
        simulated_dtype=simulated_dtype,
        bits=bits,
        signed=True,
        symmetric_output=symmetric,
        use_add=use_add,
    )

    # Make sure values are within (1 / bits) of their actual value, but only for sufficiently large elements
    np.testing.assert_allclose(
        actual_output, expected_output, rtol=1.0 / bits, atol=0.25 * expected_output.max()
    )


def test_quantized_add_or_sub_simulated():
    np.random.seed(42)
    test_cases = [
        TestCase([100], 8, True),
        TestCase([20, 4], 8, True),
        TestCase([4, 10, 3], 8, True),
        TestCase([5, 10, 2, 2], 8, True),
    ]

    for use_add in [True, False]:
        for test_case in test_cases:
            run_test_case(test_case, use_add, True)


def test_quantized_add_or_sub_not_simulated():
    np.random.seed(42)
    test_cases = [
        TestCase([100], 8, True),
        TestCase([20, 4], 8, True),
        TestCase([4, 10, 3], 8, True),
        TestCase([5, 10, 2, 2], 8, True),
    ]

    for use_add in [True, False]:
        for test_case in test_cases:
            run_test_case(test_case, use_add, False)


def test_tflite_same_io_qparams():
    data_dtype = "uint8"

    qparam_numpy = utils.AffineQParams(np.float32(0.00784314), np.int32(127), data_dtype)
    qparam_relay = utils.AffineQParams(
        relay.const(0.00784314, "float32"), relay.const(127, "int32"), data_dtype
    )

    x = relay.var("x", shape=(1, 4), dtype="float32")
    y = relay.var("y", shape=(1, 4), dtype="float32")
    z, _ = add.generate_quantized_add(
        x,
        y,
        output_qparams=qparam_relay,
        simulated_dtype=None,
        accumulation_dtype="int32",
        dequantize=False,
    )
    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_datas = [
        np.array((140, 153, 165, 178)).reshape((1, 4)),
        np.array((25, 153, 178, 216)).reshape((1, 4)),
        np.array((25, 153, 216, 165)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((204, 178, 165, 140)).reshape((1, 4)),
        np.array((204, 178, 191, 25)).reshape((1, 4)),
        np.array((204, 178, 25, 191)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((217, 204, 203, 191)).reshape((1, 4)),
        np.array((102, 204, 242, 114)).reshape((1, 4)),
        np.array((102, 204, 114, 229)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        # Data is stored in affine space, but new ops take in real space numbers
        scale_factor, zero_point, _ = qparam_numpy
        x_data = (x_datas[i] - zero_point) * scale_factor
        y_data = (y_datas[i] - zero_point) * scale_factor

        # Output should still be in affine space
        golden_output = golden_outputs[i]

        intrp = relay.create_executor("graph", device=tvm.cpu(0), target="llvm")
        op_res = intrp.evaluate(func)(x_data, y_data)
        np.testing.assert_equal(op_res.asnumpy(), golden_output)


def test_tflite_different_io_qparams():
    data_dtype = "uint8"

    qparam_numpy = utils.AffineQParams(np.float32(0.0235294), np.int32(128), data_dtype)
    qparam_relay = utils.AffineQParams(
        relay.const(0.0235294, "float32"), relay.const(128, "int32"), data_dtype
    )
    x_qparam_numpy = utils.AffineQParams(np.float32(0.0156863), np.int32(127), data_dtype)
    y_qparam_numpy = utils.AffineQParams(np.float32(0.0117647), np.int32(85), data_dtype)

    x = relay.var("x", shape=(1, 4), dtype="float32")
    y = relay.var("y", shape=(1, 4), dtype="float32")
    z, _ = add.generate_quantized_add(
        x,
        y,
        output_qparams=qparam_relay,
        simulated_dtype=None,
        accumulation_dtype="int32",
        dequantize=False,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_datas = [
        np.array((76, 140, 153, 172)).reshape((1, 4)),
        np.array((133, 140, 146, 153)).reshape((1, 4)),
        np.array((76, 140, 172, 146)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((136, 119, 128, 17)).reshape((1, 4)),
        np.array((136, 119, 111, 94)).reshape((1, 4)),
        np.array((136, 119, 17, 128)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((120, 154, 167, 124)).reshape((1, 4)),
        np.array((158, 154, 154, 150)).reshape((1, 4)),
        np.array((120, 154, 124, 163)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        # Assume the previous layer was correctly dequantized before entering our op
        x_data = (x_datas[i] - x_qparam_numpy.zero_point) * x_qparam_numpy.scale_factor
        y_data = (y_datas[i] - y_qparam_numpy.zero_point) * y_qparam_numpy.scale_factor
        golden_output = golden_outputs[i]

        intrp = relay.create_executor("graph", device=tvm.cpu(0), target="llvm")
        op_res = intrp.evaluate(func)(x_data, y_data)
        np.testing.assert_equal(op_res.asnumpy(), golden_output)
