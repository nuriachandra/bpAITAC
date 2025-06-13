#!/bin/bash
#Update job-name below
#SBATCH --job-name=218_abT
#SBATCH --partition=gpu-a40
#SBATCH --account=mostafavilab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
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

system='hyak'
exp=-1 #UPDATE
coeff=5 #UPDATE
n_filters=300 #UPDATE
model='BPcm_250'
model_num='BP218'
trial_type='complete'
batch_size=20
learning_rate=0.001
loss_fxn='PoissonLoss'
bin_size=1 # change model num too
bin_pooling='none'
scalar_head_fc_layers=1 # traditional is 1
lr_scheduler_name='Warmup'
lr_scheduler_args="{\"warmup_steps\": 1000}"
set_seed="True"

lineage_name="abT"
info_file_folder="lineage_"${lineage_name}"_complete_shallow_deprotinated_bias_quantile_normalized_4.8.25"
info_file=${info_file_folder}"/memmap/info.txt"
lineage_file="/gscratch/mostafavilab/nchand/data/ImmGen/mouse/BPprofiles1000/memmaped/"${info_file_folder}"/lineage.npy"
seq_len=998
patience=10


echo "train params: $system $coeff $exp $n_filters $model $model_num $trial_type $batch_size $learning_rate $loss_fxn $bin_size $bin_pooling $scalar_head_fc_layers $lr_scheduler_name $lr_scheduler_args $set_seed $info_file $seq_len $patience $lineage_file"
./train_script_bias_spec.sh $system $coeff $exp $n_filters $model $model_num $trial_type $batch_size $learning_rate $loss_fxn $bin_size $bin_pooling $scalar_head_fc_layers $lr_scheduler_name "$lr_scheduler_args" $set_seed $info_file $seq_len $patience $lineage_file 
