#!/bin/bash
  
# This file contains the script needed to train the AI-TAC model for mm10 mouse data
# with a specific info file
# Make sure conda ai-tac is activated before running code 
# PARAMS:
# [1] the system that it's being run on either 'hyak' or 'chelan'
# [2] the coefficient for lambda
# [3] the exponent for lambda. 2 and 3 together make lambda = coeff * 10 ^ exp
# [4] the number of filters in the body: originally was 64
# [5] the model name. current options: model='BPnetRep' or 'CNN0' or 'BPmultimax' or 'BPcm'
# [6] the model number - will be saved under this ex) BP9
# [7] 'sample' or 'complete' indicating whether sample or complete 
#       data should be used. sample is not supported on hyak
# [8] batch size ex 100
# [9] learning rate (BPnetRep used 0.001)
# [10] the loss function. Options: "PoissonNLLLoss" "MSELoss" "CompositeLoss" "CompositeLossBalanced" "CompositeLossBalancedJSD" "CompositeLossMNLL" 
# [11] the bin size 
# [12] the pooling type used for bin pooling  - either 'maxpool' or 'avgpool'
# [13] number of fully connected layers in scalar head scalar_head_fc_layers
# [14] learning rate scheduler name (e.g., "StepLR", "CosineAnnealingLR", "None")
# [15] learning rate scheduler arguments (e.g., '{"step_size": 10, "gamma": 0.1}' for StepLR)
# [16] whether the random seed should be set (true if so)
# [17] info_file
# [18] onehot seq_leq 
# [19] patience


cd ..

cluster=$1 # "hyak" or "chelan"
info_file_name=${17}

hyak_data_dir="/gscratch/mostafavilab/nchand/data/"
hyak_info_file_dir="/gscratch/mostafavilab/nchand/data/ImmGen/mouse/BPprofiles1000/memmaped/"
data_dir=$hyak_data_dir
info_file_dir=$hyak_info_file_dir
info_file=$info_file_dir$info_file_name
results_dir="/gscratch/mostafavilab/nchand/results/"
celltypes=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy"

if [ $cluster == "chelan" ]
then
    chelan_data_dir="/data/nchand/"
    chelan_info_dir="/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/"
    data_dir=$chelan_data_dir
    info_file_dir=$chelan_info_dir
    info_file=$info_file_dir$info_file_name
    results_dir="/homes/gws/nchand/MostafaviLab/results/"
    celltypes="/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy"
fi

# Override celltype path if provided as argument
if [ ! -z "${20}" ]; then
    celltypes=${20}
    echo "Using custom celltype file: $celltypes"
else
    echo "Using default celltype file: $celltypes"
fi

seq_len=${18}
batch_size=$8
# git the exponenet and coefficient for lambda which is in base 10
coeff=$2
exp=$3
bin_size=${11}
bin_pooling_type=${12}
scalar_head_fc_layers=${13}
lr_scheduler_name=${14:-"None"} # Use None as the default value
lr_scheduler_args=${15:-"None"}
set_seed=${16:-"False"}  
memmaped_data="True"
patience=${19:-10}

off_by_two="False"
if [ "$seq_len" -eq 998 ]; then
    off_by_two="True"
    echo "Sequence length is 998, setting off_by_two to true"
fi

if (($exp >= 0)); then
  lambda=$(echo "scale=0; $coeff*10^$exp" | bc)
  echo "$lambda"
else
  exp_n=${exp#-}
  lambda=$(echo "scale=20; $coeff/(10^$exp_n)" | bc)
  echo "$lambda"
fi


# UPDATE model OPTIONS: model='BPnetRep' or 'CNN0' or 'BPmultimax'
model=$5
model_num=$6

name="${model_num}_L${exp}_${coeff}" 
trial_type=$7 # either sample of complete
echo "trial type is $trial_type"
model_name=$name"/"$trial_type 

# WHERE FILES WILL BE SAVED
tstamp=$(date "+%m-%d-%Y.%H.%M")
error_dir="logs/"${model_name}"/"${tstamp}".txt"
FILE_PATH=$error_dir

# while this timestamp of a log already exists, try to make another one
while [ -f $error_dir ]
do
    echo "$error_dir still exists. Sleeping for 60 seconds"
    sleep 60
    tstamp=$(date "+%m-%d-%Y.%H.%M")
    error_dir="logs/"${model_name}"/"${tstamp}".txt"
done

mkdir -p "logs/"${model_name}
echo $error_dir

output_dir=$results_dir${model}"/"${model_name}
mkdir -p $results_dir${model}"/"${model_name}



# loss_fxn="PoissonNLLLoss"
# loss_fxn="MSELoss"
# loss_fxn="CompositeLoss"
# loss_fxn="CompositeLossBalanced"
loss_fxn=${10}
echo "loss function: $loss_fxn"
ocr_eval="False"

num_epochs="200" 
learning_rate=$9 # 0.001 was standard with BPnetRep
bias="True" # must be 'True' or 'False'
n_filters=$4 # number of filters in the model body. originally 64

echo "number of filters: $n_filters"
echo "bin size: $bin_size with bin type $bin_pooling_type"
echo "n scalar head layers: $scalar_head_fc_layers"
echo "Learning Rate Scheduler: $lr_scheduler_name"
echo "Learning Rate Scheduler Arguments: ${lr_scheduler_args}"
echo "OCR evaluation: ${ocr_eval}"
echo "Set seed: ${set_seed}"
echo "Patience: ${patience}"

start_time=$(date +%s)
echo "before if statement"
if [ "$trial_type" = "sample" ]; then

    ### FOR SMALL SAMPLE ###
    # Sample 50,000 OCR highly expressed OCRS in 6 celltypes. With bias off-by-two correction
    # celltypes=$info_file_dir"sample_bias_corrected_normalized_3.7.23/memmap/cell_names.npy"
    # info_file=$info_file_dir"sample_bias_corrected_normalized_3.7.23/memmap/info.txt" 
    # python train.py $info_file $celltypes $seq_len $name $model $output_dir $loss_fxn $num_epochs $lambda $bias $n_filters $ocr_eval $bin_size $bin_pooling_type $batch_size $learning_rate #> $error_dir 2>&1

    # sample of 100 highest expressed OCRs
    # celltypes=$info_file_dir"sample_100_bias_corrected_normalized_5.25.23/memmap/cell_names.npy"
    # info_file=$info_file_dir"sample_100_bias_corrected_normalized_5.25.23/memmap/info.txt" 
    # python train.py $info_file $celltypes $seq_len $name $model $output_dir $loss_fxn $num_epochs $lambda $bias $n_filters $ocr_eval $batch_size $learning_rate #> $error_dir 2>&1

    # random sample of 5000 all cells
    celltypes=$info_file_dir"sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/cell_names.npy"
    info_file=$info_file_dir"sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/info.txt" 
    python train.py $info_file $celltypes $seq_len $name $model $output_dir $loss_fxn $num_epochs $lambda $bias $n_filters $ocr_eval $bin_size $bin_pooling_type $scalar_head_fc_layers $batch_size $learning_rate $lr_scheduler_name "$lr_scheduler_args" $set_seed $memmaped_data --save_best_loss_model=True #> $error_dir 2>&1

    # random sample of 5000 with only 6 cells:
    # celltypes=$info_file_dir"sample_random5000_6cell_bias_corrected_normalized_5.28.23/memmap/cell_names.npy"
    # info_file=$info_file_dir"sample_random5000_6cell_bias_corrected_normalized_5.28.23/memmap/info.txt" 
    # python train.py $info_file $celltypes $seq_len $name $model $output_dir $loss_fxn $num_epochs $lambda $bias $n_filters $ocr_eval $batch_size $learning_rate #> $error_dir 2>&1

    # bottom sample of 1000
    # celltypes=$info_file_dir"sample_bottom1000_bias_corrected_normalized_5.28.23/memmap/cell_names.npy"
    # info_file=$info_file_dir"sample_bottom1000_bias_corrected_normalized_5.28.23/memmap/info.txt" 
    # python train.py $info_file $celltypes $seq_len $name $model $output_dir $loss_fxn $num_epochs $lambda $bias $n_filters $ocr_eval $batch_size #> $error_dir 2>&1

else
    # Full data with the quantile normalized counts and total counts from the OCR
    # CORRECTED FOR off-by-two
    echo "in else statement"
    python train.py $info_file $celltypes $seq_len $name $model $output_dir $loss_fxn $num_epochs $lambda $bias $n_filters $ocr_eval $bin_size $bin_pooling_type $scalar_head_fc_layers $batch_size $learning_rate $lr_scheduler_name "$lr_scheduler_args" $set_seed $memmaped_data --off_by_two=$off_by_two --patience=$patience --save_best_loss_model #> $error_dir 2>&1 
fi

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

cat << EOF  >> $output_dir"/run_details.txt"
$tstamp
$error_dir
model: $model
$name $trial_type
Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')
Loss function: $loss_fxn
${num_epochs} epochs
Bias added ${bias}
Number of filters: ${n_filters}
Learning Rate: ${learning_rate}
Learning Rate Scheduler: ${lr_scheduler_name}
Learning Rate Scheduler Arguments: ${lr_scheduler_args}
OCR evaluation: ${ocr_eval}
Set seed: ${set_seed}
Info file: ${info_file}
Off by two: ${off_by_two}
EOF

if [ "$loss_fxn" = "CompositeLoss" ] || [ "$loss_fxn" = "CompositeLossBalanced" ] || [ "$loss_fxn" = "CompositeLossBalancedJSD" ] || [ "$loss_fxn" = "CompositeLossMNLL" ]; then
cat <<- EOF  >> "$output_dir/run_details.txt"
    lambda: $lambda
EOF
fi

# add newline at the end of entry in details.txt
cat <<- EOF  >> $output_dir"/run_details.txt" 

EOF



echo Done with training script
