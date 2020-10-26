import tvm
from tvm.relay.new_quantize import Calibrater
import numpy as np

class TestCalibrater(Calibrater):
    def calibration_callback(self, var_pairs, input_subgraph_fn_pairs, output_subgraph_fn_pairs):
        output_var_map = {}
        for ((scale, zp), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(var_pairs, input_subgraph_fn_pairs):
            q_data_func = self.bind_set_variables(quantized_data_subgraph_fn)
            q_data_func = self.bind_variable(q_data_func, scale.name_hint, 0.05)
            q_data_func = self.bind_variable(q_data_func, zp.name_hint, 0)
            self.evaluate_subgraph(q_data_func, {'input': np.random.randn(1, 3, 224, 224).astype('float32')}, 'llvm', tvm.cpu())
            
            output_var_map[scale.name_hint] = 0.05
            output_var_map[zp.name_hint] = 0
            return (0.05, 0)
        

def test_calibrater():
    # Import a test graph (what test graph to use?)


    calibrater = TestCalibrater()
    calibrater.calibrate(mod, params)
