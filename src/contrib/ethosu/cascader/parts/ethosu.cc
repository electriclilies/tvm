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
#include "ethosu.h"

#include <tvm/runtime/registry.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../common.h"
#include "../stripe_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

const std::vector<int> EthosuPartNode::GetBlockShape(const StripeConfig& output_stripe_config,
                                                     bool is_rollling) {
  std::vector<int> block_shape;
  for (int axis : output_stripe_config->GetShape()) {
    block_shape.push_back(std::min(axis, 4));
  }
  return block_shape;
}

const std::vector<int> EthosuPartNode::GetBlockInputBytes_(const std::vector<int>& block_shape) {
  std::vector<int> bytes_per_input;
  std::vector<float> strides;
  std::vector<int> order;
  std::vector<int> stripes;
  std::vector<int> offset;
  for (size_t i = 0; i < block_shape.size(); i++) {
    strides.push_back(1.0);
    order.push_back(1);
    stripes.push_back(1);
    offset.push_back(0);
  }
  StripeConfig output_block_config(block_shape, block_shape, strides, order, stripes, offset);
  auto input_block_configs = CalculateInputStripeConfigs(output_block_config);
  for (const auto& input_block_config : input_block_configs) {
    bytes_per_input.push_back(mul_reduce(input_block_config->GetShape()));
  }
  return bytes_per_input;
}

const PerformanceInfo EthosuPartNode::GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                                         bool is_rolling) {
  std::vector<int> block_shape = GetBlockShape(output_stripe_config, is_rolling);
  std::vector<int> bytes_per_input = GetBlockInputBytes_(block_shape);
  int bytes_per_output = mul_reduce(block_shape);
  int num_blocks = 1;
  for (size_t i = 0; i < block_shape.size(); i++) {
    if (!is_rolling) {
      num_blocks *= output_stripe_config->GetShape()[i] * output_stripe_config->GetStripes()[i] /
                    block_shape[i];
    } else {
      num_blocks *= output_stripe_config->GetExtent()[i] / block_shape[i];
    }
  }
  int num_stripes = mul_reduce(output_stripe_config->GetStripes()) - 1;
  std::vector<size_t> read_bytes;
  for (int block_bytes : bytes_per_input) {
    read_bytes.push_back((num_blocks + num_stripes) * block_bytes);
  }
  int write_bytes = (num_blocks + num_stripes) * bytes_per_output;
  auto shape = output_stripe_config->GetShape();
  PerformanceInfo info(0, read_bytes, write_bytes);
  return info;
}

EthosuPart::EthosuPart(const TESubgraph& subgraph, const std::vector<Propagator> propagators,
                       const std::vector<int> output_quantum, int quantum_cycles) {
  auto n = make_object<EthosuPartNode>();
  ICHECK_GT(propagators.size(), 0) << "The Part must include at least one Propagator.";
  n->subgraph_ = subgraph;
  n->propagators_ = std::move(propagators);
  n->in_line_ = false;
  n->input_tensors_.resize(propagators.size());
  n->output_quantum_ = output_quantum;
  n->quantum_cycles_ = quantum_cycles;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.EthosuPart")
    .set_body_typed([](Array<te::Tensor> subgraph_inputs, te::Tensor subgraph_output,
                       Array<Propagator> propagators, Array<Integer> output_quantum,
                       int quantum_cycles) {
      std::vector<te::Tensor> vsubgraph_inputs(subgraph_inputs.begin(), subgraph_inputs.end());
      std::vector<Propagator> vpropagators(propagators.begin(), propagators.end());
      TESubgraph subgraph;
      subgraph.input_tensors = vsubgraph_inputs;
      subgraph.output_tensor = subgraph_output;
      std::vector<int> voutput_quantum = make_vector<int, Integer>(output_quantum);
      return EthosuPart(subgraph, vpropagators, voutput_quantum, quantum_cycles);
    });

TVM_REGISTER_NODE_TYPE(EthosuPartNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
