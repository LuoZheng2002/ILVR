#!/bin/bash
#SBATCH --job-name=train       # Job name
#SBATCH --output=z_train_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_train_%j.err         # Error log
#SBATCH --time=00:20:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2                     # Request 2 GPUs
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment

source activate_environment.sh
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

source run_training.sh