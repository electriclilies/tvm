from collections import namedtuple

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.python_operators import multiply, utils

TestCase = namedtuple("TestCase", ["shape", "bits", "symmetric"])
TestCase.__test__ = False


def relay_quantized_multiply(
    a_data: np.array,
    b_data: np.array,
    simulated_dtype: utils.SimulatedDTypes = utils.SimulatedDTypes.FLOAT32,
    bits: int = 8,
    signed: bool = True,
    symmetric_output=True,
):
    var_creator = utils.AffineQuantizationVarCreator()
    a = relay.var("a", shape=a_data.shape, dtype="float32")
    b = relay.var("b", shape=b_data.shape, dtype="float32")
    a_qparams, b_qparams = var_creator.get_qparams("a_qparam"), var_creator.get_qparams("b_qparam")

    output, output_qparams = multiply.generate_quantized_multiply(
        a, b, a_qparams, b_qparams, simulated_dtype=simulated_dtype, dequantize=True
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
        output,
    )

    a_qparams_actual = utils.get_quantization_parameters(
        a_data, True, bits, symmetric=symmetric_output
    )
    b_qparams_actual = utils.get_quantization_parameters(
        a_data, True, bits, symmetric=symmetric_output
    )

    mod = tvm.ir.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    f = mod["main"]
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        a_data,
        b_data,
        a_qparams_actual.scale_factor,
        a_qparams_actual.zero_point,
        b_qparams_actual.scale_factor,
        b_qparams_actual.zero_point,
    ).asnumpy()

    return result


def run_test_case(test_case: TestCase, simulated: bool):
    shape, bits, symmetric = test_case

    a = np.random.uniform(-5, 10, size=shape).astype("float32")
    b = np.random.uniform(-5, 3, size=shape).astype("float32")

    expected_output = a * b

    simulated_dtype = utils.SimulatedDTypes.FLOAT32 if simulated else None
    actual_output = relay_quantized_multiply(
        a,
        b,
        simulated_dtype=simulated_dtype,
        bits=bits,
        signed=True,
        symmetric_output=symmetric,
    )
    print(expected_output)
    print(actual_output)

    # Make sure values are within (1 / bits) of their actual value, but only for sufficiently large elements
    np.testing.assert_allclose(
        actual_output, expected_output, rtol=1.0 / bits, atol=0.25 * expected_output.max()
    )


def test_quantized_multiply_simulated():
    np.random.seed(42)
    test_cases = [
        TestCase([100], 8, True),
        TestCase([20, 4], 8, True),
        TestCase([4, 10, 3], 8, True),
        TestCase([5, 10, 2, 2], 8, True),
    ]

    for test_case in test_cases:
        run_test_case(test_case, True)


def test_quantized_multiply_not_simulated():
    np.random.seed(42)
    test_cases = [
        TestCase([100], 8, True),
        TestCase([20, 4], 8, True),
        TestCase([4, 10, 3], 8, True),
        TestCase([5, 10, 2, 2], 8, True),
    ]

    for test_case in test_cases:
        run_test_case(test_case, False)


def test_tflite_same_io_qparams():
    data_dtype = "uint8"

    lhs_scale = rhs_scale = output_scale = 0.00784314
    lhs_zero_point = rhs_zero_point = output_zero_point = 127

    x_qparam = utils.AffineQParams(
        relay.const(0.00784314, "float32"), relay.const(127, "int32"), data_dtype
    )
    y_qparam = x_qparam
    x_qparam_numpy = utils.AffineQParams(np.float32(0.00784314), np.int32(127), data_dtype)
    y_qparam_numpy = x_qparam_numpy

    x = relay.var("x", shape=(1, 4), dtype="float32")
    y = relay.var("y", shape=(1, 4), dtype="float32")
    z, _ = relay.qnn.python_operators.multiply.generate_quantized_multiply(
        x,
        y,
        x_qparam,
        y_qparam,
        simulated_dtype=None,
        dequantize=True,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_datas = [
        np.array((1, 153, 2, 178)).reshape((1, 4)),
        np.array((25, 1, 178, 216)).reshape((1, 4)),
        np.array((25, 153, 1, 165)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((204, 178, 1, 8)).reshape((1, 4)),
        np.array((204, 178, 191, 1)).reshape((1, 4)),
        np.array((204, 178, 1, 191)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        x_data = (x_datas[i] - x_qparam_numpy.zero_point) * x_qparam_numpy.scale_factor
        y_data = (y_datas[i] - y_qparam_numpy.zero_point) * y_qparam_numpy.scale_factor
        intrp = relay.create_executor("graph", device=tvm.cpu(0), target="llvm")
        op_res = intrp.evaluate(func)(x_data, y_data)

        np.testing.assert_equal(op_res.asnumpy(), (x_data * y_data).astype("float32"))


def test_tflite_different_io_qparams():
    data_dtype = "uint8"

    x_qparam = utils.AffineQParams(
        relay.const(0.0156863, "float32"), relay.const(127, "int32"), data_dtype
    )
    y_qparam = utils.AffineQParams(
        relay.const(0.0117647, "float32"), relay.const(85, "int32"), data_dtype
    )
    x_qparam_numpy = utils.AffineQParams(np.float32(0.0156863), np.int32(127), data_dtype)
    y_qparam_numpy = utils.AffineQParams(np.float32(0.0117647), np.int32(85), data_dtype)

    x = relay.var("x", shape=(1, 4), dtype="float32")
    y = relay.var("y", shape=(1, 4), dtype="float32")
    z, _ = relay.qnn.python_operators.multiply.generate_quantized_multiply(
        x,
        y,
        x_qparam,
        y_qparam,
        simulated_dtype=None,
        dequantize=True,
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

    for i in range(0, 3):
        x_data = (x_datas[i] - x_qparam_numpy.zero_point) * x_qparam_numpy.scale_factor
        y_data = (y_datas[i] - y_qparam_numpy.zero_point) * y_qparam_numpy.scale_factor
        intrp = relay.create_executor("graph", device=tvm.cpu(0), target="llvm")
        op_res = intrp.evaluate(func)(x_data, y_data)

        np.testing.assert_almost_equal(op_res.asnumpy(), (x_data * y_data).astype("float32"))
