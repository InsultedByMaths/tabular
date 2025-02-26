#!/bin/bash
#SBATCH --job-name=tabular
#SBATCH --time=12:00:00
# #SBATCH --partition=gpu
#SBATCH --partition kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --mail-user=nianli_peng@g.harvard.edu
#SBATCH -o ./runs/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./runs/errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=begin

# Initialize Conda
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate tabular

module load cuda/12.2.0-fasrc01

python train.py \
--episodes 600000 \
--solvers policy_gradient \
--learning_rate 1e-5