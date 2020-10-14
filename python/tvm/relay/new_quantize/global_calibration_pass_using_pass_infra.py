from calibration_pass import Calibrater
import numpy as np

class GlobalCalibrater(Calibrater):

    def __init__(self, scale_value, zp_value, calibration_map, mod, params=None):
        super().__init__(calibration_map, mod, params)
        self.scale_value = np.array(scale_value).astype('float32')
        self.zp_value = np.array(zp_value).astype('int32')
    
    def calibration_callback(self, scale_name, zp_name, subgraph_fn, quantized_subgraph_fn):

        #if scale_name == "add_lhs_scale_7" or scale_name == "conv2d_weight_scale_6":
        if True:
            
            #print(scale_name)
            #print(zp_name)
            #print("Subgraph")
            #print(subgraph_fn)
            #print("Quantized Subgraph")
            #print(quantized_subgraph_fn)
            
            quantized_subgraph_fn = self.bind_variable(quantized_subgraph_fn, scale_name, self.scale_value)
            quantized_subgraph_fn = self.bind_variable(quantized_subgraph_fn, zp_name, self.zp_value)
            input_np = np.random.randn(1, 3, 224, 224).astype('float32')
            
            print(self.evaluate_subgraph(quantized_subgraph_fn, {'input' : input_np}))
        return (self.scale_value, self.zp_value)