#!/bin/bash

# This script runs the prep_data file to make memmaps in the 
# and put the train, test, and validation data in the desired folder

# conda ai-tac should be activated to make this work

# 1) file path of genome one hot sequences
# 2) file path of base-pair atac-seq counts
# 3) file path of biases 
# 4) file path of peak names
# 5) path of pickled dictionary that maps peak names to chromosome name
# 6) directory where the output data will be stored and the memmaps folder will be created
hyak_data_dir="/gscratch/mostafavilab/nchand/data/"

# CHELAN 
chelan_data_dir="/data/nchand/"

data_dir=$hyak_data_dir # CHANGE WHEN SWITCH btw CHELAN/HYAK
# data_dir=$chelan_data_dir
aitac_setting='False'
cell_mask='null'
off_by_two='True'
ocr_only='False'
lineage_filepath=${data_dir}"ImmGen/mouse/immgen_lineages.csv"

cell_names_filepath=${data_dir}"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy"


cd ..

### Using Alex's naked dna bias, multi-task lineage based model### 
# selected_lineage='all'
# onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
# bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/CNNdilbiasfromcleanBACs_loglike.npz"
# dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir=$data_dir"ImmGen/mouse/BPprofiles1000/memmaped/all_lineage_complete_shallow_deprotinated_bias_quantile_normalized_4.7.25"

### Using Alex's naked dna bias, single task lineage based model### 
# selected_lineage='B'
# onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
# bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/CNNdilbiasfromcleanBACs_loglike.npz"
# dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir=$data_dir"ImmGen/mouse/BPprofiles1000/memmaped/lineage_"$selected_lineage"_complete_shallow_deprotinated_bias_quantile_normalized_4.8.25"

### Using Alex's naked dna bias, single task lineage based model, but do it for all lineages### 
lineages=('B' 'abT' 'DC' 'gdT' 'ILC' 'monocyte' 'myeloid' 'Stem&Prog' 'stroma')
onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/CNNdilbiasfromcleanBACs_loglike.npz"
dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"

for lineage in "${lineages[@]}"
do
    echo "Processing lineage: $lineage"
    out_dir=${data_dir}"ImmGen/mouse/BPprofiles1000/memmaped/lineage_"${lineage}"_complete_shallow_deprotinated_bias_quantile_normalized_4.8.25"
    
    python prep_data.py $onehot $bp_counts $bias $dict $out_dir $aitac_setting $cell_mask $off_by_two $ocr_only --lineage-filepath $lineage_filepath --selected-lineage $lineage --cell-names-filepath $cell_names_filepath
    
    echo "Completed processing for lineage: $lineage"
done

### for full data with protein free dna from BP106 model ### 
# onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
# bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/BP106_L0_1_deprotinated_mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir=$data_dir"ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_BP106_L0_1_corrected_normalized_mean_bias_pad_12.23.24"
# off_by_two='True'
# ocr_only='False'

### for full data with protein free dna from BP106 model with some weight on scalar head ### 
# onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
# bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/BP106_L9_-1_deprotinated_mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir=$data_dir"ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_BP106_L-1_9_corrected_normalized_11.6.24"
# off_by_two='True'
# ocr_only='False'

### for full data with bias trainedon closed region free dna from BP105 model
# onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
# bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
# bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/BP105_L0_1_closed_mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz" #BP105_L0_1/complete/11-02-2024.14.45
# dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir=$data_dir"ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_BP105_L0_1_corrected_normalized_12.25.24"
# off_by_two='False'
# ocr_only='False'

### for full data 

### for sample data on hyak with 100 OCRS ###
# onehot=$hyak_data_dir"mm10/sample_wBias_one_hot.npy"
# bp_counts=$hyak_data_dir"ImmGen/mouse/BPprofiles1000/sample_wBias.counts.npy"
# bias=$hyak_data_dir"ImmGen/mouse/BPprofiles1000/sample_wBias.bias.npy"
# names=$hyak_data_dir"ImmGen/mouse/BPprofiles1000/sample_wBias.names.npy"
# dict=$hyak_data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir="/gscratch/mostafavilab/nchand/bpAITAC/data_train_test/sample_normalized_250center/correct_bias"

### for sample data on hyak with 1000 OCRS
# onehot=$hyak_data_dir"mm10/sample1000_one_hot.npy"
# bp_counts=$hyak_data_dir"ImmGen/mouse/BPprofiles1000/sample1000_withBias.counts.npy"
# bias=$hyak_data_dir"ImmGen/mouse/BPprofiles1000/sample1000_withBias.bias.npy"
# names=$hyak_data_dir"ImmGen/mouse/BPprofiles1000/sample1000_withBias.names.npy"
# dict=$hyak_data_dir"ImmGen/mouse/peak_chr_dict.pkl"
# out_dir="/gscratch/mostafavilab/nchand/bpAITAC/data_train_test/sample_normalized_250center_1.17.23/"

# 10) --lineage-filepath defaults to None 
# 11) --selected-lineage defaults to None. If 'all' then a multi-task model with lineages will be selected. Otherwise should be a lineage name such as 'Stem&Prog' or 'B'
# 12) --cell-names-filepath def


python prep_data.py $onehot $bp_counts $bias $dict $out_dir $aitac_setting $cell_mask $off_by_two $ocr_only --lineage-filepath $lineage_filepath --selected-lineage $selected_lineage --cell-names-filepath $cell_names_filepath

