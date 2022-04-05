from collections import namedtuple

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.python_operators import utils


def test_get_quantization_parameters_symmetric():
    np.random.seed(42)
    shapes = [[100], [5, 15], [3, 5, 8], [1, 3, 5, 6]]

    nbits = 8
    symmetric = True
    for shape in shapes:
        arr = np.random.uniform(-10, 10, size=shape).astype("float32")
        qparam = utils.get_quantization_parameters(
            arr, True, nbits, symmetric=symmetric, as_relay=False
        )

        q_arr = arr // qparam.scale_factor + qparam.zero_point

        q_dq_arr = (q_arr - qparam.zero_point) * qparam.scale_factor

        # The quantization error should be less than the scale_factor
        np.testing.assert_allclose(arr, q_dq_arr, rtol=0, atol=qparam.scale_factor)

        # The quantization scale should cover the entire range of numbers of the array
        quantization_range = (2 ** (nbits) - 1) * qparam.scale_factor
        np.testing.assert_allclose(quantization_range, arr.max() - arr.min())
