import paddle
import os
import sys


def get_device_info(x:paddle.Tensor):
    
    return f"{x.place.custom_device_type()}:{x.place.custom_device_id()}"