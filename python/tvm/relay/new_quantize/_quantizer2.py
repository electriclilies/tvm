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

import tvm
from tvm import relay
from tvm.relay.new_quantize import Conv2DBiasAddPattern, partition_outputs, rewrite_partitions, lower_partitions, QuantizerPattern
from typing import List
import numpy as np
from tvm.contrib import graph_runtime

class Quantizer():
    def __init__(self, func, patterns: List[QuantizerPattern]): # we said List[ConcretePattern] is that actually a class
        self.patterns = patterns
        self.orig_func = func
        
        # Partition the func into sub functions containing the patterns we want to quantize
        for q_pattern in self.patterns:
            partitioned_func = q_pattern.pattern.partition(func)
        
        # Add outputs necessary for calibration, and flatten partition funcs
        tuple_subgraph_func = partition_outputs(partitioned_func)
        
        # TODO: should we lower for tuple_subgraph_mod? I'm not sure... 
        # Lower the multi-output graph to remove the partitions
        tuple_subgraph_mod = tvm.ir.IRModule()
        tuple_subgraph_mod['main'] = tuple_subgraph_func
        
        self.tuple_subgraph_mod = tuple_subgraph_mod
        
        # Rewrite the multi-output graph to be quantized, and flatten partition funcs
        outs = rewrite_partitions(self.patterns, tuple_subgraph_func)
        q_tuple_subgraph_func = outs['new_out']

        # Information about each partition used for calibration
        self.partition_infos = outs['infos_']
        
        q_tuple_subgraph_mod = tvm.ir.IRModule()
        q_tuple_subgraph_mod['main'] = q_tuple_subgraph_func 
        
        self.q_tuple_subgraph_mod = q_tuple_subgraph_mod
    