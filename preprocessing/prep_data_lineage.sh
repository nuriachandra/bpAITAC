#!/bin/bash

# This script runs the prep_data file to make memmaps in the 
# and put the train, test, and validation data in the desired folder

# 1) file path of genome one hot sequences
# 2) file path of base-pair atac-seq counts
# 3) file path of biases 
# 4) file path of peak names
# 5) path of pickled dictionary that maps peak names to chromosome name
# 6) directory where the output data will be stored and the memmaps folder will be created

aitac_setting='False'
cell_mask='null'
off_by_two='True'
ocr_only='False'
data_dir= # TODO fill in with your directory
lineage_filepath=${data_dir}"ImmGen/mouse/immgen_lineages.csv"

cell_names_filepath=${data_dir}"ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy"


cd ..

### Using naked dna bias, single task lineage based model, but do it for all lineages### 
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


python prep_data.py $onehot $bp_counts $bias $dict $out_dir $aitac_setting $cell_mask $off_by_two $ocr_only --lineage-filepath $lineage_filepath --selected-lineage $selected_lineage --cell-names-filepath $cell_names_filepath

