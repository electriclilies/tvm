
import tvm
from tvm import relay
from tvm.contrib import graph_runtime

import numpy as np
import copy

class Calibrater:
    # calibration_map is a map from relay scale and zero point variables to subgraphs. 
    # it is one of the outputs of the quantize pass. see the documentation of that pass
    # for more details
    def __init__(self, quantized_mod, calibration_map, params=None):
        self.calibration_map = calibration_map
        self.quantized_mod = quantized_mod
        self.params = params

        self.var_map = {} # map of variables to final output

    # return a tuple of (scale_value, zero_point_value) which the pass will then set in
    # the module returned from calibrate, the unquantized function wrapped in a function, and the
    # quantized subgraph wrapped in a function. You will need to bind the scale_var and zp_var for the current
    # layer before running the quantized subgraph
    def calibration_callback(self, scale_name, zp_name, data_fn, quantized_data_fn):
        raise NotImplementedError

    def op_output_callback(self, op_output_fn, quantized_op_output_fn):
        raise NotImplementedError

    # helper function to determine whether input is a weight
    def is_weight(self, expr):
        pass
    
    # bind variable name in subgraph to value (allows user to bind variable multiple times in a subgraph)
    def bind_variable(self, subgraph_fn, name, value):
        # TODO: do we have to make subgraph into a mod?
        return relay.build_module.bind_params_by_name(subgraph_fn, {name : value})
    
    # assume previous scale, zp are already bound in subgraph
    # runs the subgraph_fn passing in inputs as the inputs to the module
    def evaluate_subgraph(self, subgraph_fn, inputs):
        # TODO: add constant folding..
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            lib = relay.build(subgraph_fn, 'llvm') # TODO: what to do about params?
        module = graph_runtime.GraphModule(lib["default"](tvm.cpu())) # TODO: make the target easy to change
        module.set_input(**inputs) # TODO: make subgraphs into functions
        
        if self.params:
            module.set_input(**self.params)

        module.run()

        # subgraph only has one output # TODO: double check this is true
        return module.get_output(0).asnumpy()

    def calibrate(self):
        for (variable_pairs), (subgraph_pairs, output_pair) in self.calibration_map.items():
            #print("variable pairs: ", variable_pairs)
            #print("subgraph_pairs: ", subgraph_pairs)
            #print("output_pair: ", output_pair)
            for ((scale_var, zp_var), (subgraph_fn, quantized_subgraph_fn)) in zip(variable_pairs, subgraph_pairs):
                # bind previously set scale and zp in quantized subgraph function
                quantized_subgraph_fn = relay.build_module.bind_params_by_name(quantized_subgraph_fn, self.var_map)

                scale_name = scale_var.name_hint
                zp_name = zp_var.name_hint
                (scale_value, zp_value) = self.calibration_callback(scale_name, zp_name, subgraph_fn, quantized_subgraph_fn)

                self.var_map[scale_name] = np.array(scale_value).astype('float32')
                self.var_map[zp_name] = np.array(zp_value).astype('int32')
            
            # TODO: figure out if this is the best way to do this later -- will people ever want to set vars just using op_output_callback?
            (op_output_fn, quantized_op_output_fn) = output_pair
            self.op_output_callback(op_output_fn, quantized_op_output_fn)

        # TODO: change me to create a new mod. 
        calibrated_func = relay.build_module.bind_params_by_name(self.quantized_mod['main'], self.var_map)
        # TODO: HOW OT EXPLICITLY CONSTRUCT A MOD WITH A NAMED FUNCTION
        calibrated_mod = copy.deepcopy(self.quantized_mod)
        calibrated_mod['main'] = calibrated_func

        optimize = tvm.transform.Sequential(
            [relay.transform.FoldConstant()])
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            print("constant folding")
            calibrated_mod = optimize(calibrated_mod)
            print("done")
        
        print("calibrated mod")
        print(calibrated_mod)
        print("_____________")
        return calibrated_mod