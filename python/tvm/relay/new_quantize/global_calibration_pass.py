from tvm import relay
import numpy as np

# TODO: make this an argument to pass into calibrate
def global_calibrate(calibration_map, scale_value, zero_point_value):
    var_map = {}
    # is there a reason we store the relay var and not the name in the calibration_map?
    # given changes coming with var2id might be better to do string. 
    for (scale_var, zp_var) in calibration_map.keys():
        print(type(scale_var))
        var_map[scale_var.name_hint] = np.array(scale_value).astype('float32') #relay.const(scale_value, dtype='float32')
        var_map[zp_var.name_hint] = np.array(zero_point_value).astype('int32') #relay.const(zero_point_value, dtype='int32')
    
    return var_map

# this doesn't work because we get free vars in the output func from quantize pass then.
def calibrate(quantized_mod, calibration_map, scale_value, zero_point_value):
    prev = quantized_mod.body

    scale_zp_value_map = global_calibrate(calibration_map, scale_value, zero_point_value)
    
    for (scale, zp) in scale_zp_value_map.keys():
        prev = relay.Let(scale, calibration_map[scale], prev)
        prev = relay.Let(zp, calibration_map[zp], prev)
    print(prev)
    quantized_mod.body = prev
    
    return quantized_mod
