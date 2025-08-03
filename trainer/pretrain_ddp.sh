data_path=/home/hyg/minimind_dataset/pretrain_hq.jsonl
model_path=/home/hyg/minimind-paddle/model
python -m paddle.distributed.launch --nproc_per_node 8 pretrain.py --model_path $model_path --data_path $data_path --ddp --use_swanlab