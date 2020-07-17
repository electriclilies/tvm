/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file pad.cc
 * \brief Implementation of dynamic pad
 */
#include <topi/nn.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/op.h>

#include <vector>

#include "../../make_op.h"
#include "../../op_common.h"
#include "pad.h"

namespace tvm {
namespace relay {
namespace dyn {

// relay.dyn.nn.pad

bool PadRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const PadAttrs* param = attrs.as<PadAttrs>();
  CHECK(param != nullptr);

  // check that pad widths match lengths
  CHECK(data->shape.size() == param->pad_width.size())
      << "There should be as many pad width pairs as shape dimensions "
      << "but the shape has " << data->shape.size() << " dimensions "
      << "and there are " << param->pad_width.size() << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < param->pad_width.size(); i++) {
    CHECK(param->pad_width[i].size() == 2)
        << "Each pad width element should be a pair but at index " << i << " there are "
        << param->pad_width[i].size() << " elements.";

    auto width1 = tir::as_const_int(param->pad_width[i][0]);
    auto width2 = tir::as_const_int(param->pad_width[i][1]);
    CHECK(width1 != nullptr);
    CHECK(width2 != nullptr);

    CHECK(*width1 >= 0) << "Param width elements should be positive but first pad width at "
                        << "index " << i << " is " << *width1 << ".";
    CHECK(*width2 >= 0) << "Param width elements should be positive but first pad width at "
                        << "index " << i << " is " << *width2 << ".";

    if (!data->shape[i].as<tir::AnyNode>()) {
      auto padding = tir::make_const(data->shape[i].dtype(), *width1 + *width2);
      oshape.push_back(data->shape[i] + padding);
    } else {
      oshape.push_back(data->shape[i]);
    }
  }

  reporter->Assign(types[1], TensorType(Array<IndexExpr>(oshape), data->dtype));
  return true;
}

Array<te::Tensor> PadCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  const auto* param = attrs.as<PadAttrs>();
  CHECK(param != nullptr);

  auto pad_width = param->pad_width;
  CHECK(pad_width.size() == inputs[0].ndim() && pad_width[0].size() == 2) << "Illegal pad_width";
  Array<IndexExpr> pad_before;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_before.push_back(pad_width[i][0]);
  }
  Array<IndexExpr> pad_after;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_after.push_back(pad_width[i][1]);
  }
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return Array<te::Tensor>{topi::pad(inputs[0], pad_before, pad_after,
                                     tvm::tir::make_const(out_ttype->dtype, param->pad_value),
                                     "T_pad", topi::kElementWise, param->pad_mode)};
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakePad(Expr data, Array<Array<IndexExpr>> pad_width, double pad_value, String pad_mode) {
  auto attrs = make_object<PadAttrs>();
  attrs->pad_value = pad_value;
  attrs->pad_width = std::move(pad_width);
  attrs->pad_mode = std::move(pad_mode);
  static const Op& op = Op::Get("nn.dyn.pad");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn.dyn._make.pad").set_body_typed(MakePad);

RELAY_REGISTER_OP("nn.dyn.pad")
    .describe(R"code(Pad for n-D tensor.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<PadAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("DynamicPad", PadRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PadInferCorrectLayout<PadAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<FTVMCompute>("FTVMCompute", PadCompute);

} // dyn
} // relay
} // tvm

 // NOTE: PadInferCorrectLayout uses attrs->pad_width, so it won't work in the dynamic version. Deal with this later. 