#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --partition=root
#SBATCH -e slurm-%j.err              # File to redirect stderr
#SBATCH -o slurm-%j.out              # File to redirect stdout

#SBATCH --qos=long
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# now run your program
# srun nvidia-smi
source /scratch_root/dx522/miniconda3/etc/profile.d/conda.sh
conda activate project4

cd /scratch_root/dx522/Project4/code/SSR 
#python train.py  --data_dir ../../dataset/tiles --epoch 100 --batch_size 16 --out ./checkpoints
#python train.py  --data_dir ../../dataset/tiles --epoch 100 --batch_size 16 --out ./checkpoints_weighted
#python train.py  --data_dir ../../dataset/tiles --epochs 100 --batch_size 2 --out ./checkpoints_weighted
#python train.py  --data_dir ../../dataset/tiles --epochs 100 --batch_size 2 --out ./checkpoints_weighted --resume ./checkpoints_weighted/best_awan_syn.pth
python test.py

## Get the number of cores
#CORES=$(nproc)

## Launch your process on each core
#for ((i=1; i<=$CORES; i++)); do
#    # Replace the command below with your actual process
#    (echo "Process $i started"; sleep 30; echo "Process $i finished") &
#done
#wait
#sleep 30
