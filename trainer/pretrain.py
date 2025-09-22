import os
import sys
__package__ = "trainer"
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
from paddle.distributed import fleet
from contextlib import nullcontext
from paddlenlp.transformers import AutoTokenizer
from model.modeling_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.custom_dataset import PretrainDataset
from utils.get_custom_device import auto_detect_device
import paddle.nn.functional as F
from utils.log_info import log_training_and_model_info
from utils.callback import ProgressHandler
from paddle.distributed.utils.log_utils import get_logger
warnings.filterwarnings('ignore')

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, swanlab, opt):
    world_size = dist.get_world_size() if args.ddp else 1
    per_process_batches = len(train_loader)
    per_process_opt_steps = math.ceil(per_process_batches / args.accumulation_steps)
    global_opt_steps_per_epoch = per_process_opt_steps * world_size
    is_main = (not args.ddp) or (dist.get_rank() == 0)
    callback = ProgressHandler(total_steps=global_opt_steps_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}", enabled=is_main)

    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    
    step_times = 0

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        opt.set_lr(lr)

        ctx = paddle.amp.auto_cast(args.dtype in ['float16', 'bfloat16'], dtype=args.dtype) if args.dtype in ['float16', 'bfloat16'] and args.device != 'cpu' else nullcontext()

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.reshape([-1, res.logits.shape[-1]]),
                Y.reshape([-1])
            ).reshape(Y.shape)
            loss = (loss * loss_mask.astype('float32')).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()

            opt.clear_grad()
            step_times += 1

            if (step_times * world_size) % args.log_interval == 0:

                callback.update(n=(dist.get_world_size() if args.ddp else 1) * step_times)
                step_times = 0

        if (swanlab is not None) and (not args.ddp or dist.get_rank() == 0):
            swanlab.log({"loss": loss.item() * args.accumulation_steps,
                        "lr": opt.get_lr()  
                        })

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}_swanlab.pth'

            state_dict = model.state_dict()
            state_dict = {k: v.astype('float16') for k, v in state_dict.items()}  
            paddle.save(state_dict, ckp)
            model.train()

if __name__ == "__main__":
    
    defualt_device = auto_detect_device()

    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--model_path", type=str, default='./model')
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default=defualt_device)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = defualt_device

    base_seed = 1337
    paddle.seed(base_seed)

    args.swanlab_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MiniMindForCausalLM(lm_config)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = None

    if args.ddp:
        
        ddp_rank = dist.get_rank()
        paddle.seed(base_seed + ddp_rank)
        fleet.init(is_collective=True)
        model = fleet.distributed_model(model)
        train_sampler = DistributedBatchSampler(train_ds, batch_size=args.batch_size)
        train_loader = DataLoader(
            train_ds,
            num_workers=args.num_workers,
            batch_sampler=train_sampler
        )

    else:
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            batch_sampler=train_sampler
        )


    opt = optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters(), grad_clip=nn.ClipGradByGlobalNorm(args.grad_clip) if args.grad_clip > 0 else None)

    if args.ddp:
        opt = fleet.distributed_optimizer(opt)



    scaler = paddle.amp.GradScaler(enable=(args.dtype in ['float16', 'bfloat16']))


    if args.use_swanlab and (not args.ddp or ddp_rank == 0):
        import swanlab

        swanlab.init(
            project=args.swanlab_project, 
            name=args.swanlab_run_name,
            config={
                "learning_rate": args.learning_rate,
                "epochs":args.epochs
            }
        )
    else:
        swanlab = None


    iter_per_epoch = len(train_loader)

    logger = get_logger("INFO","minimind")
    world_size = dist.get_world_size() if args.ddp else 1
    log_training_and_model_info(logger, args, lm_config, model, world_size)
    for epoch in range(args.epochs):
        train_epoch(epoch, swanlab, opt)