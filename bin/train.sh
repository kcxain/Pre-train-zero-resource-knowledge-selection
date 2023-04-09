#!/bin/bash
#SBATCH -J wow_8
#SBATCH -o wow_8.out
#SBATCH -e wow_8.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH --mem=18GB
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1

model_name=bert-base-uncased
checkpoint=models/stage1/stage1_32_24_best_cpt
params_file=src/configs/params.json

python train.py \
        --model_name_or_path ${model_name} \
        --checkpoint ${checkpoint} \
        --params_file ${params_file}