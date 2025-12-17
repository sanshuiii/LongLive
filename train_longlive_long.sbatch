#!/bin/bash
#SBATCH --job-name=longlive-long
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=8                    # 多机：按需修改节点数
#SBATCH --ntasks-per-node=1          # 每节点仅启 1 个 launcher 任务
#SBATCH --gpus-per-node=8            # 每节点 GPU 数
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/long/%x-%j.out      # 建议添加日志输出目录
#SBATCH --error=logs/long/%x-%j.err

echo "=== Job started on $(date) ==="
echo "Running on nodes:"
echo $SLURM_NODELIST
echo "Using $SLURM_NNODES nodes"

##############################
# Environment Setup
##############################

# 加载你的 MI/Slurm 环境
source ~/slurm_tools/mi.sh

# 激活 conda 环境
conda activate longlive

# 建议查看 python / torch cuda 是否正确
echo "Python: $(which python)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# Project path and config
CONFIG=configs/longlive_train_long.yaml
LOGDIR=logs/long
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-one-logger

echo "=== Job finished at $(date) ==="
