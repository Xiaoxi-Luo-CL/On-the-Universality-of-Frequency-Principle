#!/bin/bash

#SBATCH --job-name=gpt-spectral
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --partition=COMPLING
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err


echo "spectral bias"
for data_option in uniform sphered random clustered
do
    for target_func in 1 2 3
    do
        /u501/x25luo/.conda/envs/spectral/bin/python experiments.py --lr 0.0002 --data_option ${data_option} --target_func ${target_func} --steps 20000 
    done
done