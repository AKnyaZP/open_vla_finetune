#!/bin/bash

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/jovyan/shares/SR006.nfs2/tensorflow_datasets/\
  --dataset_name dsynth_atomic_tasks \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapters \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 500 
