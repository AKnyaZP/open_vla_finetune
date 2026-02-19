#!/bin/bash
set -ex

CONDA_ENV=/mnt/virtual_ai0001053-01202_SR006-nfs1/.mlspace/envs/knyazev_venv
PYTHON=$CONDA_ENV/bin/python3.10
WORKDIR=/mnt/virtual_ai0001053-01202_SR006-nfs2/knyazev/openvla

cd $WORKDIR

$PYTHON -m pip install -e .

$PYTHON -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/virtual_ai0001053-01202_SR006-nfs2/tensorflow_datasets/ \
  --dataset_name dsynth_atomic_tasks \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapters \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 500 
