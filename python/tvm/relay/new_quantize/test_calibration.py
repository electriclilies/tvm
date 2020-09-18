import tvm
from tvm.relay.new_quantize import Calibrater, quantize_pass
from tvm.relay.testing import resnet
import numpy as np

class TestCalibrater(Calibrater):
    def calibration_callback(self, var_pairs, input_subgraph_fn_pairs, output_subgraph_fn_pair):
        value_dict = {}
        for ((scale, zp), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(var_pairs, input_subgraph_fn_pairs):
            data_func = self.bind_set_variables(data_subgraph_fn)
            self.evaluate_subgraph(data_func, {'input': np.random.randn(1, 3, 224, 224).astype('float32')}, 'llvm', tvm.cpu()) #TODO: what is input for this

            q_data_func = self.bind_set_variables(quantized_data_subgraph_fn)
            q_data_func = self.bind_variable(q_data_func, scale.name_hint, 0.05)
            q_data_func = self.bind_variable(q_data_func, zp.name_hint, 0)
            self.evaluate_subgraph(q_data_func, {'input': np.random.randn(1, 3, 224, 224).astype('float32')}, 'llvm', tvm.cpu())
            
            value_dict[scale.name_hint] = 0.05
            value_dict[zp.name_hint] = 0
        
        output_subgraph_fn = output_subgraph_fn_pair[0]
        quantized_output_subgraph_fn = output_subgraph_fn_pair[1]

        output_fn = self.bind_set_variables(output_subgraph_fn)
        self.evaluate_subgraph(output_fn, {'input': np.random.randn(1, 3, 224, 224).astype('float32')}, 'llvm', tvm.cpu())

        quantized_output_fn = self.bind_set_variables(quantized_output_subgraph_fn)
        quantized_output_fn = self.bind_variables(quantized_output_fn, value_dict)

        self.evaluate_subgraph(quantized_output_fn, {'input': np.random.randn(1, 3, 224, 224).astype('float32')}, 'llvm', tvm.cpu())
        return value_dict

# This test makes sure that the calibration pass runs properly
def test_calibrater():
    #TODO: change this to a smaller network!!
    mod, params = resnet.get_workload(batch_size=1)
    quantized_mod, calibration_map = quantize_pass.quantize(mod, params, skip_layers=[])

    test_calibrater = TestCalibrater()
    calibrated_mod = test_calibrater.calibrate(quantized_mod, calibration_map, params)

if __name__ == '__main__':
    test_calibrater()