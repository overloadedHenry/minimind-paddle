import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle import optimizer
from paddle.io import DataLoader, DistributedBatchSampler
from contextlib import nullcontext
from paddlenlp.transformers import AutoTokenizer
from model.modeling_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.custom_dataset import PretrainDataset



def check_model_parameters(model):
    print("Model parameters info:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, requires_grad: {param.stop_gradient}")
        if param.grad is not None:
            print(f"  grad shape: {param.grad.shape}")
    print("-" * 50)

if __name__ == '__main__':

    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)

    model = MiniMindForCausalLM.from_pretrained(lm_config).to('npu:0')
    tokenizer = AutoTokenizer.from_pretrained('/home/hyg/minimind-paddle/model')
    check_model_parameters(model)

    