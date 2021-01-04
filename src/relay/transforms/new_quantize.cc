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
 * \file src/tvm/relay/transforms/new_quantize.cc
 * \brief Relay Quantization related passes
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../ir/dataflow_matcher.h"

namespace tvm {
namespace relay {
namespace quantize {

struct CalibrationInfo {
  Op op; // might not need?
  Attrs attrs; // What type should attrs be?

  Array<Array<Var>> input_scale_zps;
  Array<int> input_idxs;
  int output_idx;

  // Should I store the scales and zps for the inputs in here?
};

class PartitionOutputs : public MixedModeMutator {
 public:
  Expr GetPartitionOutputs(const Expr& expr) {
    new_outputs.clear();
    if (auto func = expr.as<FunctionNode>()) {
      new_outputs.push_back(func->body);
    } else {
      new_outputs.push_back(expr);
    }
    VisitExpr(expr);
    Expr out;
    if (auto func = expr.as<FunctionNode>()) {
      out = Function(func->params, Tuple(new_outputs), Type{}, Array<TypeVar>{}, func->attrs);
    } else {
      out = Tuple(new_outputs);
    }
    return out;
  }
 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) { 
    auto* post_node = post.as<CallNode>();
    ICHECK(post_node != nullptr);
    if (auto* func_node = post_node->op.as<FunctionNode>()) {
      if (func_node->attrs.defined() && func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
        for(const auto& arg : post_node->args) {
          new_outputs.push_back(arg);
        }
        new_outputs.push_back(post);
      }
    }
    return post;
  }
  
  Array<Expr> new_outputs;
};

class RewritePartitions : public MixedModeMutator {
 public:
  RewritePartitions(const DFPatternCallback& callback) : callback_(callback) {}

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) { 
    auto* post_node = post.as<CallNode>();
    ICHECK(post_node != nullptr);
    if (auto* func_node = post_node->op.as<FunctionNode>()) {
      if (func_node->attrs.defined() && func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
          auto matcher = DFPatternMatcher(func_node->body);
          if (matcher.Match(callback_->pattern, func_node->body)) {
          Array<Var> params = func_node->params;
          Array<Expr> call_args= post_node->args;
          Expr new_body = callback_->function(pre->op.as<FunctionNode>()->body, func_node->body, matcher.GetMemo());
          // Add new parameters to the function arguments
          for (const auto& param : FreeVars(new_body)) {
            if (std::find(params.begin(), params.end(), param) == params.end()) {
              params.push_back(param);
              call_args.push_back(Var(param->name_hint(), param->type_annotation));
            }
          }
          Expr new_func = Function(params, new_body, Type{}, Array<TypeVar>{}, func_node->attrs);
          return Call(new_func, call_args, Attrs{}, Array<Type>{});
        }
      }
    }
    return post;
  }
  DFPatternCallback callback_;
};

TVM_REGISTER_GLOBAL("relay.new_quantize.partition_outputs")
    .set_body_typed([](const Expr& expr) { return PartitionOutputs().GetPartitionOutputs(expr); });
TVM_REGISTER_GLOBAL("relay.new_quantize.rewrite_partitions")
    .set_body_typed([](const DFPatternCallback& callback, const Expr& expr) { return RewritePartitions(callback).Mutate(expr); });
}  // namespace quantize
}  // namespace relay
}  // namespace tvm
