#!/bin/bash
#SBATCH -J s1_1288_wow
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 3-00:00:00
#SBATCH --mem=18GB
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1

python stage1.py --model_name_or_path "../models/stage1/stage1_12_8_8_best.cpt" \
                 --do_train \
                 --evaluation_strategy steps \
                 --eval_steps 500 \
                 --save_steps 500 \
                 --logging_steps 500 \
                 --load_best_model_at_end \
                 --metric_for_best_model accuracy \
                 --num_train_epochs 5 \
                 --max_seq_length 512 \
                 --doc_stride 128 \
                 --cache_dir "../cache" \
                 --output_dir "../stage1/samples_12_8_8_wow_dev/" \
                 --overwrite_output_dir --per_device_train_batch_size 4 \
                 --gradient_accumulation_steps 8 \
                 --warmup_steps 1000 \
                 --weight_decay 0.01 \
                 --max_answer_length 90 \
                 --save_total_limit 5 \
                 --learning_rate 5e-6