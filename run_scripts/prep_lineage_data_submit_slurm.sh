#!/bin/bash
#Update job-name below
#SBATCH --job-name=prep_lineage
#SBATCH --partition=gpu-l40s 
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nchand@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

conda activate ai-tac
git_branch='main' # update
git checkout $git_branch 
git rev-parse --abbrev-ref HEAD # print the current branch

./prep_data_lineage.sh > prep_lineage_data_log.txt 2>&1
