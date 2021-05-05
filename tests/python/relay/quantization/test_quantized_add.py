from collections import namedtuple

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.python_operators import add, utils

TestCase = namedtuple("TestCase", ["shape", "bits", "symmetric", "simulated"])
TestCase.__test__ = False


def relay_quantized_add_or_sub(
    a_data: np.array,
    b_data: np.array,
    simulated: utils.SimulatedDTypes = utils.SimulatedDTypes.FLOAT32,
    bits: int = 8,
    signed: bool = True,
    symmetric_output=True,
    use_add: bool = True,
):
    actual_output_qparams = utils.get_quantization_parameters(
        a_data + b_data, True, bits, symmetric=symmetric_output
    )

    var_creator = utils.AffineQuantizationVarCreator()
    a = relay.var("a")
    b = relay.var("b")
    output_qparams = var_creator.get_qparams("output")

    if use_add:
        output, output_qparams = add.generate_quantized_add(
            a, b, output_qparams, simulated=simulated, dequantize=True
        )
    else:
        output, output_qparams = add.generate_quantized_sub(
            a, b, output_qparams, simulated=simulated, dequantize=True
        )

    f = relay.Function(
        [a, b, output_qparams.scale_factor, output_qparams.zero_point],
        output,
    )

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        a_data, b_data, actual_output_qparams.scale_factor, actual_output_qparams.zero_point
    ).asnumpy()

    return result


def run_test_case(test_case: TestCase, use_add: bool):
    shape, bits, symmetric, simulated = test_case

    a = np.random.uniform(-5, 10, size=shape).astype("float32")
    b = np.random.uniform(-5, 3, size=shape).astype("float32")

    if use_add:
        expected_output = a + b
    else:
        expected_output = a - b

    simulated_type = utils.SimulatedDTypes.FLOAT32 if simulated else None
    actual_output = relay_quantized_add_or_sub(
        a,
        b,
        simulated=simulated_type,
        bits=bits,
        signed=True,
        symmetric_output=symmetric,
        use_add=use_add,
    )

    # Make sure values are within (1 / bits) of their actual value, but only for sufficiently large elements
    np.testing.assert_allclose(
        actual_output, expected_output, rtol=1.0 / bits, atol=0.25 * expected_output.max()
    )


def test_quantized_add_or_sub():
    np.random.seed(42)
    test_cases = [
        TestCase([100], 8, True, True),
        TestCase([20, 4], 8, True, True),
        TestCase([4, 10, 3], 8, True, True),
        TestCase([5, 10, 2, 2], 8, True, True),
    ]

    for use_add in [True, False]:
        for test_case in test_cases:
            run_test_case(test_case, use_add)
