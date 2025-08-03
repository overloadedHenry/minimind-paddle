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
from dataset.custom_dataset import SFTDataset
from utils.get_custom_device import auto_detect_device
warnings.filterwarnings('ignore')


def Logger(content):
    if not args.ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, swanlab):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # for param_group in optimizer._param_groups:
        #     param_group['lr'] = paddle.assign(lr)
        opt.set_lr(lr)

        if ctx_dict["auto_cast"]:
            ctx = paddle.amp.auto_cast(ctx_dict["dtype"])
        else:
            ctx = nullcontext()

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
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(param.name, param.grad.shape)
            # check_model_parameters(model)
            # exit()
            # paddle.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            opt.clear_grad()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    opt.get_lr(),
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (swanlab is not None) and (not args.ddp or dist.get_rank() == 0):
                swanlab.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": opt.get_lr(),
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}_swanlab.pth'

            # if isinstance(model, paddle.DataParallel):
            #     state_dict = model.state_dict()
            # else:
            #     state_dict = model.state_dict()
            state_dict = model.state_dict()
            # state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            paddle.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model')
    model = MiniMindForCausalLM(lm_config)

    return model, tokenizer



def init_distributed_mode():
    if not args.ddp: return
    global ddp_rank, DEVICE

    ddp_rank = dist.get_rank()
    ddp_world_size = dist.get_world_size()
    DEVICE = f"npu:{ddp_rank}"




# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    
    defualt_device = auto_detect_device()
    # print(defualt_device)
    # exit()
    parser = argparse.ArgumentParser(description="MiniMind SFT")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--checkpoint", type=str, default="/home/hyg/minimind-paddle/out/pretrain_512_0_swanlab.pth")
    parser.add_argument("--out_dir", type=str, default="./out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
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
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="/home/hyg/minimind_dataset/sft_512.jsonl")
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(state_dict)

    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if not p.stop_gradient) / 1e6:.3f} 百万')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedBatchSampler(train_ds, batch_size=args.batch_size) if args.ddp else None

    if args.ddp:

        init_distributed_mode()
        
        paddle.seed(base_seed + ddp_rank)

        fleet.init(is_collective=True)
        model = fleet.distributed_model(model)
        train_loader = DataLoader(
            train_ds,
            num_workers=args.num_workers,
            batch_sampler=train_sampler
        )
        print("DDP setting up")

    else:
        model = model.to(args.device)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            batch_sampler=train_sampler
        )

    opt = optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters(), grad_clip=nn.ClipGradByGlobalNorm(args.grad_clip) if args.grad_clip > 0 else None)



    global ctx_dict
    ctx_dict = {
        'auto_cast': False,
        'dtype': None
    }

    # ctx = nullcontext() if device_type == "cpu" else paddle.amp.auto_cast()
    if args.dtype == 'bfloat16' and paddle.device.is_compiled_with_cinn():
        ctx_dict['auto_cast'] = True
        ctx_dict['dtype'] = 'bfloat16'
        
    else:
        dtype = 'float16' if args.dtype == 'float16' else 'float32'
        if dtype != 'float32':
            ctx_dict['auto_cast'] = True
            ctx_dict['dtype'] = dtype
        else:
            ctx_dict['auto_cast'] = False


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
    for epoch in range(args.epochs):
        train_epoch(epoch, swanlab)