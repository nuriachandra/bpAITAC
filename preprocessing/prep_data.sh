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

data_dir= "/data/nchand/" # TODO FILL in
aitac_setting='False'
cell_mask='null'
off_by_two='True'

# TODO update these with your file names
### for full data with bias trained on closed region free dna from BP105 model
onehot=$data_dir"mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz"
bp_counts=$data_dir"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz"
bias=$data_dir"ImmGen/mouse/BPprofiles1000/bias/BP105_L0_1_closed_mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz" #BP105_L0_1/complete/11-02-2024.14.45
dict=$data_dir"ImmGen/mouse/peak_chr_dict.pkl"
out_dir=$data_dir"ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_complex_closedregions_BP105_L0_1_quantile_normalized_4.4.25"
off_by_two='False'
ocr_only='False'

python prep_data.py $onehot $bp_counts $bias $dict $out_dir $aitac_setting $cell_mask $off_by_two $ocr_only

