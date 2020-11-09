import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import ExprVisitor
from tvm.relay.frontend.common import infer_type

import numpy as np
import copy

def _bind_params(func, params): # TODO: replace me with relay.bind_params
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.expr.const(v)
    return relay.expr.bind(func, bind_dict)

class Calibrater:
    # calibration_map is a map from relay scale and zero point variables to subgraphs. 
    # it is one of the outputs of the quantize pass. see the documentation of that pass
    # for more details
    def __init__(self):
        self.calibration_map = None
        self.quantized_mod = None
        self.params = None

        self.var_map = {} # map of variables to final calibration value

    # For an op in the original graph with 2 inputs, var_pairs is a 3d tuple of the form 
    # (((input1_scale, input1_zp), (input2_scale, input2_zp))).
    # The corresponding input_subgraph_pairs are 
    # (((input1_data_fn, input1_quantized_data_fn), (input2_data_fn, input2_quantized_data_fn)))
    # where input1_data_fn is the original, unquantized version of input1 in runnable function form, and
    # input1_quantized_data_fn is the quantized version of input1 in runnable function form. 
    # The output_subgraph_pair is (output_data_fn, output_quantized_data_fn)
    # output_data_fn is the original, unquantized version of the operation in runnable function form, and
    # output_quantized_data_fn is the dequantized output of the quantized operation
    # You will need to pass scales and zero points to output_quantized_data_fn for all the scales and zero points
    # corresponding to this node (ie, all the scale and zero point variables in var_pairs)
    def calibration_callback(self, var_pairs, input_subgraph_fn_pairs, output_subgraph_fn_pair):
        raise NotImplementedError

    def constant_calibration_callback(self, var_pairs, input_value_pairs, output_value_pair):
        raise NotImplementedError

    # helper function to determine whether input is a weight
    # this will be kind of repetitive, maybe we can
    def is_weight(self, expr):

        class WeightVisitor(ExprVisitor):
            def __init__(self):
                super().__init__()
                self.is_weight = False
            
            def visit_call(self, call):
                if (call.op.num_inputs == 1):
                    self.visit(call.args[0])
            
            def visit_constant(self, constant):
                self.is_weight = True
        
        weightvisitor = WeightVisitor()
        weightvisitor.visit(expr)

        return weightvisitor.is_weight

    
    # Bind variables that we set in previous callbacks
    def bind_set_variables(self, subgraph_fn):
        return relay.build_module.bind_params_by_name(subgraph_fn, self.var_map)

    # Bind variable name to value in subgraph_fn (allows user to bind variable multiple times in a subgraph)
    def bind_variable(self, subgraph_fn, name, value):
        return relay.build_module.bind_params_by_name(subgraph_fn, {name : value})

    # Binds a map of variable names to values in subgraph_function
    def bind_variables(self, subgraph_fn, var_map):
        return relay.build_module.bind_params_by_name(subgraph_fn, var_map)
    
    # assume previous scale, zp are already bound in subgraph
    # runs the subgraph_fn passing in inputs as the inputs to the module
    def evaluate_subgraph(self, subgraph_mod, inputs):
        # TODO: Throw a readable error if user has not set a lot of vars in var_map
        if self.var_map:
            subgraph_mod.set_input(**self.var_map) # TODO: make sure this is OK if var map contains things not in the inputs
        subgraph_mod.set_input(**inputs) # TODO: assert that this doesnt create any weird behavior
        subgraph_mod.run()
        # subgraph only has one output # TODO: double check this is true
        return subgraph_mod.get_output(0).asnumpy()

    # TODO: move mod, calibration_map, params to inputs here
    def calibrate(self, quantized_mod, calibration_map, params=None):
        self.quantized_mod = quantized_mod
        self.calibration_map = calibration_map
        self.params = params
        self.var_map = {}
        for (variable_pairs), (input_subgraph_pairs, output_subgraph_pair) in self.calibration_map.items():
            value_dict = self.calibration_callback(variable_pairs, input_subgraph_pairs, output_subgraph_pair)
            self.var_map.update(value_dict) # Merge value_dict into self.var_map

        calibrated_func = _bind_params(self.quantized_mod['main'], self.var_map)
        calibrated_mod = tvm.ir.IRModule()
        calibrated_mod['main'] = calibrated_func
        
        optimize = tvm.transform.Sequential(
            [relay.transform.FoldConstant()])
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            calibrated_mod = optimize(calibrated_mod)
        
        return calibrated_mod