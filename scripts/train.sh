#!/bin/bash

#SBATCH --job-name=train_dr_spaam

#SBATCH --output=/home/hu/Projects/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=huzjkevin@gmail.com

#SBATCH --partition=lopri

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --gres=gpu:1

#SBATCH --time=0-20:00:00

#SBATCH --signal=TERM@120

WS_DIR="$HOME/Projects/planar_optical_flow"
SCRIPT="train_dr_spaam_cluster.py"

cd ${WS_DIR}

# wandb on

srun --unbuffered python ${SCRIPT} --cfg ./config/dr_spaam.yaml
