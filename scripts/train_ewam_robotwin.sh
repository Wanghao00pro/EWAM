#!/bin/bash
# EWAM training launcher — Stage-2 finetune (RoboTwin dim16, reweighted loss).
#
# Entry point : train/train_ewam.py  (model.loss_reweight: true selects the
#               reweighted loss — see configs/ewam_robotwin.yaml)
# Config      : configs/ewam_robotwin.yaml
# DeepSpeed   : configs/zero1.json (ZeRO stage 1, bf16)
#
# Usage:
#   bash scripts/train_ewam_robotwin.sh
#   # Override GPUs / port:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29101 bash train_ewam_robotwin.sh
#   # Multi-node (torchrun):
#   NNODES=2 RANK=0 MASTER_ADDR=<ip> bash train_ewam_robotwin.sh

# cd to the package root (this script lives in scripts/)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} torchrun \
    --nnodes=${NNODES:-1} \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-29100} \
    train/train_ewam.py \
    --deepspeed configs/zero1.json \
    --config configs/ewam_robotwin.yaml
