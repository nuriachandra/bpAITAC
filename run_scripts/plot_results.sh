#!/bin/bash


cd ..

# in hyak:
# observed_tcounts_path="/gscratch/mostafavilab/nchand/bpAITAC/data_train_test/complete_normalized_1.17.23/memmap/val.total_counts.dat"
# model_dir="/gscratch/mostafavilab/nchand/results/BPnetRep"

type="complete"

# in chelan:
### SAMPLE
# data_info_file="/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/info.txt"
# n_filters=300
# batch_size=20
# model_dir="/homes/gws/nchand/MostafaviLab/results/BPcm"
# cell_names="/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/cell_names.npy"
# peak_names="/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/val.names.dat"
# seq_len="998"
# models=("BP57_20_L0_0.5/sample")
# timestamps=("04-17-2024.18.42")

# /data/nchand/analysis/BP6_L-12/04-17-2023.21.31/val_correlation.txt
### COMPLETE
data_info_file="/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt"
n_filters=300
batch_size=20
model_dir="/data/nchand/analysis"
cell_names="/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy"
peak_names="/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/val.names.dat"
seq_len="998"

# models=("BP2_L1" "BP2_L3" "BP2_L10" "BP3_L1" "BP3_L3" "BP3_L10")
# timestamps=("02-15-2023.07.59" "02-15-2023.00.32" "02-15-2023.00.30" "02-15-2023.00.19" "02-15-2023.09.09" "02-15-2023.00.21")

# models=("BP4_L1" "BP4_L0" "BP4_L-10" "BP4_L-3" "BP4_L10")
# timestamps=("03-07-2023.21.27" "03-07-2023.20.01" "03-07-2023.20.52" "03-08-2023.08.14" "03-08-2023.10.27")
# models=("BP4_L0" "BP4_L-10" "BP4_L-11" "BP4_L-12" "BP4_L-13" "BP4_L-14" "BP4_L1")
# timestamps=("04-11-2023.14.14" "03-18-2023.10.09" "04-03-2023.16.52" "04-09-2023.12.38" "04-09-2023.12.40" "04-11-2023.14.19" "03-18-2023.10.22")

# models=("BP6_L-11" "BP6_L-1")
# timestamps=("04-18-2023.01.56" "04-18-2023.15.51")

# models=("BPcm/BP17_L0_0" "BPcm/BP17_L-1_7")
# timestamps=("09-22-2023.08.38" "05-26-2023.14.26")

# models=("BPcm/BP17_L0_0")
# timestamps=("05-26-2023.00.18")

models=("BPcm/BP60_20_L-1_5" "BPcm/BP56_20_L-1_8")
timestamps=("complete/04-22-2024.17.19" "complete/04-22-2024.07.10")
analyzed_data_files=("/data/nchand/analysis/BPcm/BP60_20_L-1_5/complete/04-22-2024.17.19/analysis.npz" "/data/nchand/analysis/BPcm/BP56_20_L-1_8/complete/04-22-2024.07.10/analysis.npz")

# some more models with small profile use
# models=("BP2_L0" "BP2_L-1" "BP2_L-2" "BP3_L0" "BP3_L-1" "BP3_L-2")
# timestamps=("02-16-2023.21.31" "02-16-2023.22.09" "02-16-2023.22.10" "02-16-2023.21.22" "02-16-2023.22.12" "02-16-2023.22.13")

for i in ${!models[@]}; do
    dir=$model_dir/${models[$i]}/${timestamps[$i]}
    title=${models[$i]}"_"${timestamps[$i]}
    model_name=${models[$i]}
    analyzed_data_file=${analyzed_data_files[$i]}
    echo $title
    echo $dir
    python plot_results.py --model_directory $dir \
                           --data_info_file $data_info_file \
                           --model_title $title \
                           --n_filters $n_filters \
                           --cell_names $cell_names \
                           --peak_names $peak_names \
                           --seq_len $seq_len \
                           --batch_size $batch_size \
                           --model_name $model_name \
                           --example_profiles \
                            --example_profiles_eval_set "Validation" \
                            --analyzed_data_file $analyzed_data_file\
                            --example_profiles_ocr_celltypes "B.Fo.Sp" "B.GC.CB.Sp" "NK.27+11b-.Sp" "GN.Thio.PC"\
                            --example_profiles_compare_celltypes "B.Fo.Sp" "B.GC.CB.Sp" "T.DP.Th" "NK.27+11b-.Sp" "GN.Thio.PC"\
                            # --save_predictions \
                            # --run_analysis \
done