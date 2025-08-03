
python -m paddle.distributed.launch --nproc_per_node 8 trainer/train_full_sft.py  --ddp --use_swanlab