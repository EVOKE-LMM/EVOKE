#!/bin/bash

# 从命令行获取 checkpoint 参数
checkpoint=$1

# 如果没有提供 checkpoint 参数，则报错并退出
if [ -z "$checkpoint" ]; then
    echo "Error: Please provide the checkpoint path as an argument."
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

for ds in "qwen_vl_eval_vqa"; do
    python -m torch.distributed.launch --use-env \
        --nproc_per_node=${NPROC_PER_NODE:-1} \
        --nnodes=${WORLD_SIZE:-1} \
        --node_rank=${RANK:-0} \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-12345} \
        evaluate_vqa.py \
        --checkpoint "$checkpoint" \
        --dataset "$ds" \
        --batch-size 2 \
        --num-workers 2
done
