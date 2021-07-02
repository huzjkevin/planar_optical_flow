#!/bin/bash

#SBATCH --job-name=train_obj_det

#SBATCH --output=/home/hu/Projects/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=huzjkevin@gmail.com

#SBATCH --partition=lopri

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --gres=gpu:2

#SBATCH --time=0-20:00:00

#SBATCH --signal=TERM@120

#SBATCH --array=1-4

WS_DIR="$HOME/Projects/planar_optical_flow"
SCRIPT="train_obj_det.py"

cd ${WS_DIR}

# wandb on

file=`ls ./config/config_obj_det.yaml | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

srun --unbuffered python ${SCRIPT} --cfg $file
