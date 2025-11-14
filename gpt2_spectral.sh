#!/bin/bash

#SBATCH --job-name=gpt-spectral
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --partition=COMPLING
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

#SBATCH -o JOB-%j.out
#SBATCH -e JOB-%j.err


echo "GPT2 spectral bias"
/u501/x25luo/.conda/envs/spectral/bin/python gpt2_4gram_next_token.py