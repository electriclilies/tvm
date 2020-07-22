# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return, invalid-name, unused-argument, too-many-arguments, consider-using-in
"""Backend compiler related feature registration"""

from __future__ import absolute_import

from topi.util import get_const_tuple

from tvm.runtime import convert
from tvm.te.hybrid import script
from ...op import register_shape_func, register_compute
from ...op import register_injective_schedule, register_broadcast_schedule
from .._nn import _pad_shape_func
import tvm
import topi

# upsampling
@register_compute("nn.dyn.upsampling")
def compute_upsampling(attrs, inputs, out_dtype):
    print("compute_upsampling called")
    data = inputs[0]
    scale_h = inputs[1][0]
    scale_w = inputs[2][0]
    layout = attrs.layout
    method = attrs.method
    align_corners = attrs.align_corners
    return [topi.nn.upsampling(data, scale_h, scale_w, layout, method, align_corners)]

register_injective_schedule("nn.dyn.upsampling")

# pad
register_broadcast_schedule("nn.dyn.pad")

#####################
#  Shape functions  #
#####################

# upsampling

@script
def _upsampling_nhwc_shape_func(dshape, scale_h, scale_w):
    out = output_tensor((4,), "int64")
    batch_size = dshape.shape[0]
    in_height = dshape.shape[1]
    in_width = dshape.shape[2]
    channels = dshape.shape[3]
    out[0] = int64(batch_size)
    out[1] = int64(round(in_height * h_scale[0]))
    out[2] = int64(round(in_width * w_scale[0]))
    out[3] = int64(channels)
    return out

@script
def _upsampling_nchw_shape_func(dshape, scale_h, scale_w):
        out = output_tensor((4,), "int64")
        batch_size = dshape.shape[0]
        channels = dshape.shape[1]
        in_height = dshape.shape[2]
        in_width = dshape.shape[3]
        out[0] = int64(batch_size)
        out[1] = int64(channels)
        out[2] = int64(round(in_height * scale_h[0]))
        out[3] = int64(round(in_width * scale_w[0]))
        return out

@register_shape_func("nn.dyn.upsampling", True)
def upsampling_shape_func(attrs, inputs, _):
    print(inputs[0].shape)
    print(inputs[1])
    print(inputs[2])
    if (attrs.layout == "NHWC"):
        return [_upsampling_nhwc_shape_func(inputs[0], inputs[1], inputs[2])]
    if (attrs.layout == "NCHW"):
        return [_upsampling_nchw_shape_func(inputs[0], inputs[1], inputs[2])]

@script
def _dyn_pad_shape_func(data, pad_width):
    out = output_tensor((data.shape[0],), "int64")
    for i in const_range(out.shape[0]):
        out[i] = pad_width[i, 0] + pad_width[i, 1] + data.shape[i]
    return out

@register_shape_func("nn.dyn.pad", True)
def pad_shape_func(attrs, inputs, data):
    """
    Shape function for dynamic pad op.
    """
    print(inputs[0].shape[0])
    print(inputs[1])
    return [_dyn_pad_shape_func(inputs[0], inputs[1])]
