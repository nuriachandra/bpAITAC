#!/bin/bash
#Update job-name below
#SBATCH --job-name=65L-1_7
#SBATCH --partition=gpu-a40
#SBATCH --account=mostafavilab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nchand@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

conda activate ai-tac
git_branch='learning-rate-scheduler' # update
git checkout $git_branch 
git rev-parse --abbrev-ref HEAD # print the current branch

system='hyak'
exp=-1 #UPDATE
coeff=7 #UPDATE
n_filters=300 #UPDATE
model='BPcm'
model_num='BP65'
trial_type='complete'
batch_size=20
learning_rate=0.0001
loss_fxn='CompositeLossMNLL'
bin_size=1 # change model num too
bin_pooling='none'
scalar_head_fc_layers=1 # traditional is 1
lr_scheduler_name='WarmupCosineDecayLR'
lr_scheduler_args="{\"T_max\": 20, \"warmup_steps\": 1000}"

echo "train params: $system $coeff $exp $n_filters $model $model_num $trial_type $batch_size $learning_rate $loss_fxn $bin_size $bin_pooling $scalar_head_fc_layers $lr_scheduler_name $lr_scheduler_args"
./train_script.sh $system $coeff $exp $n_filters $model $model_num $trial_type $batch_size $learning_rate $loss_fxn $bin_size $bin_pooling $scalar_head_fc_layers $lr_scheduler_name "$lr_scheduler_args"
