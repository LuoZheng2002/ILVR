#!/bin/bash
set -euo pipefail
#SBATCH --job-name=smoke-train        # Job name
#SBATCH --output=z_smoke_%j.out       # Standard output log (%j = job ID)
#SBATCH --error=z_smoke_%j.err        # Error log
#SBATCH --time=00:20:00               # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
#SBATCH --gres=gpu:1                  # Request 1 GPU for smoke run
#SBATCH --cpus-per-task=4             # CPU cores
#SBATCH --mem=64G                     # RAM
#SBATCH --account=bfdz-delta-gpu      # Slurm account/project

source activate_environment.sh

export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=0
export TRAIN_IMAGE_ROOT=data
export TEST_IMAGE_ROOT=

MASTER_PORT=${MASTER_PORT:-29510}

torchrun --standalone --nproc_per_node=1 --master_port="${MASTER_PORT}" src/main.py \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --epochs 0.02 \
  --task "zebra-cot" \
  --gradient_accumulation_steps 1 \
  --stage stage1 \
  --smoke_test_mode \
  --max_seq_length_stage1 2048 \
  --warm_up_steps 1 \
  --data_path "data/TRAIN.jsonl" \
  --log_file "smoke_train.log" \
  --latent_size 8 \
  --ce_weight 1 \
  --save_model_path "checkpoints_smoke" \
  --cache_dir "${HF_HOME}" \
  --mixed_precision bf16 \
  --ema_tau 0 \
  --save_steps 10

echo "smoke training finished"
