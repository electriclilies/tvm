from calibration_pass import Calibrater
import numpy as np
import tvm

class GlobalCalibrater(Calibrater):

    def __init__(self, scale_value, zp_value, weight_scale_value, weight_zp_value, mod, calibration_map, params=None):
        super().__init__(mod, calibration_map, params)
        self.scale_value = np.array(scale_value).astype('float32')
        self.zp_value = np.array(zp_value).astype('int32')
        self.weight_scale_value = np.array(weight_scale_value).astype('float32')
        self.weight_zp_value = np.array(weight_zp_value).astype('int32')
    
    # join these 2 callbacks
    def calibration_callback(self, scale_name, zp_name, subgraph_fn, quantized_subgraph_fn):
        #input_np = np.random.randn(1, 3, 224, 224).astype('float32')
        
        #print(self.evaluate_subgraph(quantized_subgraph_fn, {'input' : input_np}))
        #print(self.evaluate_subgraph(subgraph_fn, {'input' : input_np}))
        if "weight" in scale_name:
            return (self.weight_scale_value, self.weight_zp_value)
        return (self.scale_value, self.zp_value)
    
    def op_output_callback(self, op_output_fn, quantized_op_output_fn):
        pass

class TestCalibrater(Calibrater):
    def __init__(self, calibration_map, mod, params=None):
        super().__init__(calibration_map, mod, params)
        self.scale_value = 0
        self.zp_value = 0
    
    def calibration_callback(self, scale_name, zp_name, subgraph_fn, quantized_subgraph_fn):
        self.scale_value = self.scale_value + 1
        self.zp_value = self.zp_value + 1
        return (self.scale_value, self.zp_value)
    
    def op_output_callback(self, op_output_fn, quantized_op_output_fn):
        pass
