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

class PatternCalibrationInfoNode : public Object { // Change name later
 public:
  DFPattern pattern; 

  Array<Array<Var>> input_scale_zps;
  Array<Integer> input_idxs;
  Integer output_idx;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("input_scale_zps", &input_scale_zps);
    v->Visit("input_idxs", &input_idxs);
    v->Visit("output_idx", &output_idx);
  }

  static constexpr const char* _type_key = "PatternCalibrationInfoNode";
  TVM_DECLARE_BASE_OBJECT_INFO(PatternCalibrationInfoNode, Object);
};


class PatternCalibrationInfo : public ObjectRef {
 public:
  TVM_DLL PatternCalibrationInfo(DFPattern pattern, Array<Array<Var>> input_scale_zps, Array<Integer> input_idxs, Integer output_idx);
  TVM_DEFINE_OBJECT_REF_METHODS(PatternCalibrationInfo, ObjectRef, PatternCalibrationInfoNode);
};

PatternCalibrationInfo::PatternCalibrationInfo(DFPattern pattern, Array<Array<Var>> input_scale_zps, Array<Integer> input_idxs, Integer output_idx) {
  ObjectPtr<PatternCalibrationInfoNode> n = make_object<PatternCalibrationInfoNode>();
  n->pattern = std::move(pattern);
  n->input_scale_zps = std::move(input_scale_zps);
  n->input_idxs = std::move(input_idxs);
  n->output_idx = std::move(output_idx);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternCalibrationInfoNode);

TVM_REGISTER_GLOBAL("relay.new_quantize.PatternCalibrationInfo")
    .set_body_typed([](DFPattern pattern, Array<Array<Var>> input_scale_zps, Array<Integer> input_idxs, Integer output_idx) {
      return PatternCalibrationInfo(pattern, input_scale_zps, input_idxs, output_idx);
    });

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

class RewritePartitions : protected MixedModeMutator {
 public:
  RewritePartitions(const Array<DFPatternCallback>& callbacks) : callbacks_(callbacks) {}
  Expr Rewrite (const Expr& expr) {
    // Preprocessing
    if (auto* func = expr.as<FunctionNode>()) {
      if (auto* tuple = func->body.as<TupleNode>()) {
        orig_outputs_ = tuple->fields;
      } else {
        orig_outputs_.push_back(func->body); 
      }
    } else {
      if (auto* tuple = expr.as<TupleNode>()) {
        orig_outputs_ = tuple->fields;
      } else {
        orig_outputs_.push_back(expr); 
      }
    }
    Expr new_out = MixedModeMutator::Mutate(expr);
    // Postprocessin

    return new_out;//{new_out, infos_};
  }
 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) { 
    // Cast the post as a call node and assert it actually is a call
    auto* post_node = post.as<CallNode>();
    ICHECK(post_node != nullptr);
    // Check to see if the Call is calling a Function
    if (auto* func_node = post_node->op.as<FunctionNode>()) {
      // If it's calling a function, check to see if it has attributes that it's a been partitioned from a Pattern
      if (func_node->attrs.defined() && func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
        // If this is a pattern function, create a matcher on it's body
        auto matcher = DFPatternMatcher(func_node->body);
        // Find the callback that matches this pattern
        for (const auto& callback : callbacks_) {
          if (matcher.Match(callback->pattern, func_node->body)) {
            // extract the current params and call-level args
            Array<Var> params = func_node->params;
            Array<Expr> call_args = post_node->args;

            //PatternCalibrationInfo info(callback->pattern, );
            Array<Integer> input_idx;
            // Get the indices of the arguments to this function in the output tuple
            for (auto arg : pre->args) {
              auto itr = std::find(orig_outputs_.begin(), orig_outputs_.end(), arg);
              ICHECK(itr == orig_outputs_.end()) << "Didn't find the arugment in the output tuple. Indicates a possible problem in PartitionOutputs. "
              input_idx.push_back(std::distance(orig_outputs_.begin(), itr));
            }
            // Get the index of the output of this function 
            auto itr = std::find(orig_outputs_.begin(), orig_outputs_.end(), pre);
            ICHECK(itr == orig_outputs_.end()) << "Didn't find the output in the output tuple. Indicates a possible problem in PartitionOutputs. ")
            Integer output_idx(std::distance(orig_outputs_.begin(), itr));

            // FIND THE SCALE / ZPS
 

            /*outputs = [x, w, b, x2, w2, b2]
            input_idx x -> 0
            input_idx w -> 1
            ...*/
            // create a new body based on the callback
            Expr new_body = callback->function(pre->op.as<FunctionNode>()->body, func_node->body, matcher.GetMemo());
            // find parameters added to the new body that weren't there before
            // find all of the free variables in the new body
            for (const auto& param : FreeVars(new_body)) {
              // check to see if that free variable is in the old parameter list
              if (std::find(params.begin(), params.end(), param) == params.end()) {
                // if not, add it to the new parameter list
                std::cout << param << std::endl;
                params.push_back(param);
                // Create a new call-level arg for it
                call_args.push_back(Var(param->name_hint(), param->type_annotation));
                // Make that new arg an input to the top-level function
                new_params_.push_back(call_args.back());
              }
            }
            // Create a new function with new params and body
            Expr new_func = Function(params, new_body, Type{}, Array<TypeVar>{}, func_node->attrs);
            // Call the new function with the new args
            return Call(new_func, call_args, Attrs{}, Array<Type>{});
          }
        }
      }
    }
    return post;
  }
  Array<DFPatternCallback> callbacks_;
  Array<PatternCalibrationInfo> infos_;
  Array<Expr> new_params_;
  Array<Expr> orig_outputs_;
};

TVM_REGISTER_GLOBAL("relay.new_quantize.partition_outputs")
    .set_body_typed([](const Expr& expr) { return PartitionOutputs().GetPartitionOutputs(expr); });
TVM_REGISTER_GLOBAL("relay.new_quantize.rewrite_partitions")
    .set_body_typed([](const Array<DFPatternCallback>& callbacks, const Expr& expr) { return RewritePartitions(callbacks).Rewrite(expr); });
}  // namespace quantize
}  // namespace relay
}  // namespace tvm
