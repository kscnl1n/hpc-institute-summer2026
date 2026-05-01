#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --output=llm-train-%j.out
#SBATCH --error=llm-train-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

module load apptainer

apptainer run --nv llm_train.sif
