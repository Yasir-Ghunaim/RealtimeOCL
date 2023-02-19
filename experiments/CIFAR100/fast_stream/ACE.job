#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ACE
#SBATCH -o /path/to/output.%J.out
#SBATCH -e /path/to/error.%J.err
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6


# activate the conda environment
module purge
module load gcc/11.1.0
conda activate realtime_ocl

# run the application:
cd ../../..
python main.py \
--dataset 'cifar100' \
--batch_size 10 \
--lr 0.001 \
--lr_type 'constant' \
--batch_delay 0 \
--gradient_steps 1 \
--output_dir '/path/to/tensorboard/output' \
--workers 4 \
--method 'ACE' \
--seed 123 \
--size_replay_buffer 100