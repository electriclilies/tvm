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
 * \file src/relay/op/call_dps.cc
 * \brief call_dps operator
 */

#include <tvm/relay/op.h>
#include <tvm/relay/attrs/annotation.h>

namespace tvm {
namespace relay {

/* relay.call_dps */
bool CallDPSRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    // types: [func, arg_tuple, out_tuple]
    ICHECK_EQ(types.size(), 3);

    // TODO: Can we check that the func is a primfunc here as well as just a PrimExpr?

    const auto* func = types[0].as<FuncTypeNode>();
    if (func == nullptr) {
        ICHECK(types[0].as<IncompleteTypeNode>())
        << "call_dps: expect second input type to be TupleType but get " << types[0];
        return false;
    }

    const auto* args = types[1].as<TupleTypeNode>();
    if (args == nullptr) {
        ICHECK(types[1].as<IncompleteTypeNode>())
        << "call_dps: expect second input type to be TupleType but get " << types[1];
        return false;
    }

    const auto* outs = types[2].as<TupleTypeNode>();
    if (outs == nullptr) {
        ICHECK(types[2].as<IncompleteTypeNode>())
        << "call_dps: expect third type to be TupleType but get " << types[2];
    return false;
    }

    // Assign the output shape of call_dps to be the shape of the destination tensor
    // TODO: I think this might work for dynamic shapes too. Not sure
    reporter->Assign(types[3], types[2]);
    return true;
}

Expr MakeCallDPS(Expr func, Expr arguments, Expr outputs) {
    static const Op& op = Op::Get("vm.invoke_tvm_op");
    return Call(op, {func, arguments, outputs}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.call_dps").set_body_typed(MakeCallDPS);


RELAY_REGISTER_OP("call_dps")
    .describe(R"code(Call a TIR function in destination passing style.

- **func**: The TIR function (PrimFunc).
- **arugments**: Arguments to be passed to the PrimFunc, packed together in a Tuple.
- **outputs**: Outputs for the PrimFunc to write to, packed together in a Tuple.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("func", "PrimFunc", "The function to call.")
    .add_argument("arguments", "Tuple", "Arguments to call func on.")
    .add_argument("outputs", "Tuple", "Tensors where the PrimFunc will write output.")
    .set_support_level(3)
    .add_type_rel("CallDPSRel", CallDPSRel);

/* relay.call_tir */

bool CallTIRRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    // types: [func, arg_tuple, ret_type]
    ICHECK_EQ(types.size(), 3);

    const auto* func = types[0].as<FuncTypeNode>();
    if (func == nullptr) {
        ICHECK(types[0].as<IncompleteTypeNode>())
        << "call_tir: expect second input type to be TupleType but get " << types[0];
        return false;
    }

    const auto* args = types[1].as<TupleTypeNode>();
    if (args == nullptr) {
        ICHECK(types[1].as<IncompleteTypeNode>())
        << "call_tir: expect second input type to be TupleType but get " << types[1];
        return false;
    }

    reporter->Assign(types[2], func->ret_type); // Can this also work in dyn case? I think so
    return true;
}

Expr MakeCallTIR(Expr func, Expr args, Attrs attrs) {
    // attrs are always call_tir attrs.
    // Might be able to do the is dynamic check here... though is departure from the separation of dyn and static ops that matt and I did
    return Call(func, {args}, attrs);
}

TVM_REGISTER_GLOBAL("relay.op._make.call_tir").set_body_typed(MakeCallDPS);


RELAY_REGISTER_OP("call_tir")
    .describe(R"code(Call a TIR function in destination passing style.

- **func**: The TIR function (PrimFunc).
- **arugments**: Arguments to be passed to the PrimFunc, packed together in a Tuple.
- **outputs**: Outputs for the PrimFunc to write to, packed together in a Tuple.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("func", "PrimFunc", "The function to call.")
    .add_argument("args", "Tuple", "Tuple containing arguments to the func.")
    .set_support_level(3)
    .add_type_rel("CallTIRRel", CallTIRRel);

} // namespace relay
} // namespace tvm