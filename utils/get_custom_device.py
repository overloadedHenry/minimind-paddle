import paddle
import os
import sys


def get_device_info(x:paddle.Tensor):
    
    return f"{x.place.custom_device_type()}:{x.place.custom_device_id()}"\
    

def auto_detect_device():

    if paddle.is_compiled_with_custom_device('npu'):
        return "npu"  
    
    if paddle.is_compiled_with_cuda():
        return "gpu"  
    
    return "cpu"