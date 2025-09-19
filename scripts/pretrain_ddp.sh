data_path=/home/hyg/minimind_dataset/pretrain_hq.jsonl
model_path=/home/hyg/minimind-paddle/model
export ASCEND_RT_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --nproc_per_node 1 pretrain.py --model_path $model_path --data_path $data_path --use_swanlab