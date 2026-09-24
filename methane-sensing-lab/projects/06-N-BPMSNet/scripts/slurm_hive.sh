#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=model
#SBATCH --partition=tide #dgxl_long,dgxl_epic,hive_epic,hive_prob
#SBATCH --qos=long #dgxl_long_high #,dgxl_epic_low,hive_epic_high,hive_long_maxtres_gpu2

#SBATCH -e slurm-%j.err              # File to redirect stderr
#SBATCH -o slurm-%j.out              # File to redirect stdout
#SBATCH --mem=100G                   # Memory per processor
#SBATCH --time=24:00:00              # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of tasks
##SBATCH --ntasks-per-socket=1       # Maximum number of tasks on each socket
#SBATCH --cpus-per-task=1            # CPUs per task
#SBATCH --gres=gpu:1                 # Number of GPUs

# This is where the actual work is done.

#nvidia-smi
source ./miniconda3/etc/profile.d/conda.sh
conda activate project3

module load cuda

#export TMPDIR=~/tmp
#mkdir -p ~/tmp

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Own
cd ./N_BPMSNet
python train_single.py
python train.py
python test.py
python test_single.py

