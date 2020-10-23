import tvm
from tvm.relay.new_quantize import Calibrater
import numpy as np

class GlobalCalibrater(Calibrater):

    def __init__(self, scale_value, zp_value, weight_scale_value, weight_zp_value):
        super().__init__()
        self.scale_value = np.array(scale_value).astype('float32')
        self.zp_value = np.array(zp_value).astype('int32')
        self.weight_scale_value = np.array(weight_scale_value).astype('float32')
        self.weight_zp_value = np.array(weight_zp_value).astype('int32')
    
    def calibration_callback(self, var_pairs, input_subgraph_fn_pairs, output_subgraph_fn_pair):
        output_values = []
        for ((scale, zp), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(var_pairs, input_subgraph_fn_pairs):
            
            print(quantized_data_subgraph_fn)
            print(scale)
            print(zp)
            q_data_subgraph = self.bind_variable(quantized_data_subgraph_fn, scale.name_hint, 2.0)
            # for some reason, bind_params_by_name will not set things sequentially
            print("set scale")
            print(q_data_subgraph)
            q_data_subgraph = self.bind_variable(quantized_data_subgraph_fn, zp.name_hint, 0)
            print("set zp")
            print(q_data_subgraph)
            self.evaluate_subgraph(q_data_subgraph, [np.random.randn(1, 3, 224, 224).astype('float32')], 'llvm', tvm.cpu())
            exit()
            if self.is_weight(data_subgraph_fn):
                scale_value = self.weight_scale_value
                zp_value = self.weight_zp_value
            else:
                scale_value = self.scale_value
                zp_value = self.zp_value

            output_values.append((scale_value, zp_value))
        return tuple(output_values)
