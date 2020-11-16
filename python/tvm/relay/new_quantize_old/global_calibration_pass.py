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
        value_dict = {} # dictionary from scale, zp name to value
        for ((scale_var, zp_var), (data_subgraph_fn, quantized_data_subgraph_fn)) in zip(var_pairs, input_subgraph_fn_pairs):

            if self.is_weight(data_subgraph_fn):
                value_dict[scale_var.name_hint] = self.weight_scale_value
                value_dict[zp_var.name_hint] = self.weight_zp_value
            else:
                value_dict[scale_var.name_hint] = self.scale_value
                value_dict[zp_var.name_hint] = self.zp_value

        return value_dict
