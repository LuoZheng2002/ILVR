#!/bin/bash
#SBATCH --job-name=sim2real1       # Job name
#SBATCH --output=z_sim2real1_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_sim2real1_%j.err         # Error log
#SBATCH --time=02:00:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
# #SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment

# Go to your project directory
# cd /u/zluo8/qwen
cd /projects/bfdz/zluo8/sim2real
source activate_environment.sh
cd /projects/bfdz/zluo8/sim2real
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

python acebench.py --model-name zhuiguang-ning/LoopTool-8B --user-api-model-name gpt-4.1 --num-gpus 1