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
from contextlib import nullcontext
from paddlenlp.transformers import AutoTokenizer, PretrainedTokenizerFast
from model.modeling_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.custom_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # for param_group in optimizer._param_groups:
        #     param_group['lr'] = paddle.assign(lr)
        optimizer.set_lr(lr)

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
            scaler.unscale_(optimizer)
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(param.name, param.grad.shape)
            paddle.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.clear_grad()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.get_lr(),
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, paddle.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            paddle.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if not p.stop_gradient) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"gpu:{ddp_local_rank}"
    paddle.device.set_device(DEVICE)

def auto_detect_device():
    """自动检测可用设备，优先级：NPU > GPU > CPU"""
    # 检查NPU（华为昇腾）
    if paddle.is_compiled_with_custom_device('npu'):
        # 通过环境变量判断是否分配了NPU设备
        # if os.environ.get('ASCEND_VISIBLE_DEVICES') is not None:
        return "npu"  # 返回第一个NPU设备
    
    # 检查GPU
    if paddle.is_compiled_with_cuda():
        # gpu_count = paddle.device.cuda.get_device_count()
        # if gpu_count > 0:
        return "gpu"  # 返回第一个GPU设备
    
    # 默认回退CPU
    return "cpu"

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    
    defualt_device = auto_detect_device()
    # print(defualt_device)
    # exit()
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--model_path", type=str, default='./model')
    parser.add_argument("--out_dir", type=str, default="./out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default=defualt_device)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
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
    parser.add_argument("--data_path", type=str, default="E:\paddle_llm\minimind_data\pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = defualt_device

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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
        # ctx = paddle.amp.auto_cast(dtype=dtype) if dtype != 'float32' else nullcontext()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, device_type

    base_seed = 1337
    paddle.seed(base_seed)


    if ddp:
        init_distributed_mode()
        args.device = paddle.device.set_device(DEVICE)
        rank = dist.get_rank()
        paddle.seed(base_seed + rank)


    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedBatchSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        batch_sampler=train_sampler
    )

    scaler = paddle.amp.GradScaler(enable=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters())

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = paddle.DataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)