data_path=/data/gehongyu/minimind-dataset/pretrain_hq.jsonl
model_path=/data/gehongyu/minimind-paddle/model
# export ASCEND_RT_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1,2
export NCCL_P2P_DISABLE=1

python -m paddle.distributed.launch --nproc_per_node 1 ./trainer/pretrain.py --model_path $model_path --data_path $data_path --epochs 3 --batch_size 32 --use_moe True --ddp
# python -m debugpy --listen 5678 --wait-for-client ./trainer/pretrain.py --model_path $model_path --data_path $data_path  --batch_size 8 --use_moe True

# python  ./trainer/pretrain.py --model_path $model_path --data_path $data_path  --batch_size 8 --use_moe True
