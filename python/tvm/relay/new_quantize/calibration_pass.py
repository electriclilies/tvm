
import tvm
from tvm import relay
from tvm.contrib import graph_runtime

import numpy as np

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
    def calibration_callback(self, scale_name, zp_name, subgraph_fn, quantized_subgraph_fn):
        raise NotImplementedError

    # is the start of this a constant? (dominator)
    def is_weight(self, subgraph):
        raise NotImplementedError
    
    # bind variable name in subgraph to value (allows user to bind variable multiple times in a subgraph)
    def bind_variable(self, subgraph_fn, name, value):
        # TODO: do we have to make subgraph into a mod?
        return relay.build_module.bind_params_by_name(subgraph_fn, {name : value})
    
    # assume previous scale, zp are already bound in subgraph
    def evaluate_subgraph(self, subgraph_fn, input):
        # TODO: how do I turn this into a CHECK
        print(subgraph_fn)
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(subgraph_fn, 'llvm', params=self.params) # TODO: what to do about params?
        module = graph_runtime.GraphModule(lib["default"](tvm.cpu())) # TODO: make the target easy to change
        

    def calibrate(self):
        for (scale_var, zp_var), (subgraph, quantized_subgraph) in self.calibration_map.items():
            
            # bind parameters whose scales were set in previous passes to quantized subgraph
            fn = relay.Function(relay.analysis.free_vars(subgraph), subgraph)
            q_fn = relay.Function(relay.analysis.free_vars(quantized_subgraph), quantized_subgraph)
            
            # bind previously set scale and zp in quantized subgraph function
            q_fn = relay.build_module.bind_params_by_name(q_fn, self.var_map)

            scale_name = scale_var.name_hint
            zp_name = zp_var.name_hint
            (scale_value, zp_value) = self.calibration_callback(scale_name, zp_name, fn, q_fn)

            self.var_map[scale_name] = np.array(scale_value).astype('float32')
            self.var_map[zp_name] = np.array(zp_value).astype('int32')

        self.quantized_mod['main'] = relay.build_module.bind_params_by_name(self.quantized_mod['main'],
                                                                            self.var_map)
        return self.quantized_mod