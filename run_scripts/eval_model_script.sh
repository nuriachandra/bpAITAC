#!/bin/bash

# example run: ./eval_model_script.sh /data/nchand/analysis/BPnetRep/BP71 BPnetRep validation
# /homes/gws/nchand/MostafaviLab/results/BPcm/BP107_seed_L-1_5/complete/11-08-2024.12.16
cd ..
run_type='complete'

if [ "$run_type" == 'sample' ]; then # NOTE I DON"T THINK THIS ACTUALLY WORKS YET WILL APPEAR AS COMPLETE WITHOUT CLEAR DIFFERENTIATOR
    chelan_info_file='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/info.txt'
else
    # chelan_info_file='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt'
    # chelan_info_file='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_BP106_L0_1_corrected_normalized_mean_bias_pad_12.23.24/memmap/info.txt' # BPcm protein free bias model
    # chelan_info_file='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_BP105_L0_1_corrected_normalized_12.25.24/memmap/info.txt'
    # chelan_info_file='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/all_lineage_complete_bias_corrected_normalized_12.30.24/memmap/info.txt' # lineage data old
    chelan_info_file='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_shallow_deprotinated_bias_quantile_normalized_4.1.25/memmap/info.txt'
fi

# Set the base directory
base_dir="$1"
# Set the type of model
model_type="$2"

eval_set="$3"

# Set the bin size (default to 1 if not provided)
bin_size="${4:-1}"

seq_len="${5:-998}"


n_celltypes="${6:-90}"

model_filename="${7:-best_model}" 
save_file_name=$eval_set"_analysis.npz"

if [ "$7" != "" ]; then
    echo "new filename"
    # Only append the model_filename if it was explicitly passed
    save_file_name=${save_file_name}"_"${model_filename}
fi

off_by_two="False"
if [ "$seq_len" -eq 998 ]; then
    off_by_two="True"
    echo "Sequence length is 998, setting off_by_two to true"
fi


# Find all "best_model" files with paths starting with the base directory
find "$base_dir"* -type f -name "$model_filename" | while read -r best_model; do
    # Extract the directory path from the file path
    directory=$(dirname "$best_model")
    echo "$directory"
    python eval_model.py --saved_model_path $best_model --infofile_path $chelan_info_file --output_dir $directory --get_scalar_corr --get_profile_corr --get_jsd --model_type $model_type --eval_set $eval_set --bin_size $bin_size --seq_len $seq_len --off_by_two $off_by_two --n_celltypes $n_celltypes --save_file_name $save_file_name #--ocr_start 0 --ocr_end 250
done

