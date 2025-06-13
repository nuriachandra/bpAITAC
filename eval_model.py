"""
This file is designed to be used to evaluate models that have
already been trainined

Arguments: 

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from load_model import model_analysis_from_saved_model, get_predictions, load_model, load_data
from plot_utils_bpaitac import histogram
import os
from BPcm import BPcm
from BPcm_250 import BPcm_250
from BPbi import BPbi
from BPbi_shallow import BPbi_shallow
from BPnetRep import BPnetRep
from BPol import BPol
from BPcm_skinny import BPcm_skinny
from BPmp import BPmp
from BPcm_super_skinny import BPcm_super_skinny
from BPcm_bias0 import BPcm_bias0


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze a saved model.')
    parser.add_argument('--saved_model_path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--n_celltypes', type=int, default=90,
                        help='Number of cell types')
    parser.add_argument('--n_filters', type=int, default=300,
                        help='Number of filters')
    parser.add_argument('--infofile_path', type=str, required=True,
                        help='Path to the info file')
    parser.add_argument('--get_scalar_corr', action='store_true',
                        help='Compute scalar correlation')
    parser.add_argument('--get_profile_corr', action='store_true',
                        help='Compute profile correlation')
    parser.add_argument('--get_jsd', action='store_true',
                        help='Compute JSD')
    parser.add_argument('--eval_set', type=str, default='validation',
                        choices=['validation', 'training', 'testing'],
                        help='Evaluation set (validation or training or testing)')
    parser.add_argument('--bin_size', type=int, default=1,
                        help='Bin size')
    parser.add_argument('--bin_pooling_type', type=str, default='MaxPool1d',
                        choices=['MaxPool1d', 'AvgPool1d'],
                        help='Bin pooling type')
    parser.add_argument('--scalar_head_fc_layers', type=int, default=1,
                        help='Number of fully connected layers in the scalar head')
    parser.add_argument('--ocr_start', type=int, default=375,
                        help='Start position of OCR')
    parser.add_argument('--ocr_end', type=int, default=625,
                        help='End position of OCR')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for the metrics and plot')
    parser.add_argument('--model_type', type=str, default='BPcm', 
                        help='The type of the model structure to be used. Currently only supports BPcm and BPnetRep')
    parser.add_argument('--seq_len', type=int, default=998)
    parser.add_argument('--off_by_two', type=str, default="True")
    parser.add_argument('--get_complete_corr_metrics', action='store_true', default=False)
    parser.add_argument('--save_file_name', type=str, default=None)
    return parser.parse_args()

def save_metrics(metrics, output_dir, eval_set, save_file_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if save_file_name == None:
        output_file = os.path.join(output_dir, eval_set+'_analysis.npz')
    else:
        output_file = os.path.join(output_dir, save_file_name)
    print("output file path", output_file)
    np.savez_compressed(output_file, **metrics)

def plot_top_ocr_profile_corr(top_ocr_profile_corr, output_dir, eval_set):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, eval_set+"_top_ocr_profile_corr.png")
    histogram(top_ocr_profile_corr, x_label='Top OCR Profile Correlation', title='Distribution of Top OCR Profile Correlation', format='png', output_path=output_file)

def plot_top_jsd(top_ocr_jsd, output_dir, eval_set):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, eval_set+"_top_ocr_jsd.png")
    histogram(top_ocr_jsd, x_label='Top OCR JSD', title='Distribution of Top OCR JSD', format='png', output_path=output_file)

def plot_scalar_corr(scalar_corr, output_dir, eval_set):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, eval_set+"_scalar_corr.png")
    histogram(scalar_corr, x_label='Total Count Pearson Correlation', title='Distribution of Regional Accessibility Correlation', format='png', output_path=output_file)
    
def eval_model(saved_model_path, n_celltypes, n_filters, infofile_path,
               get_scalar_corr, get_profile_corr, get_jsd, eval_set,
               bin_size=1, bin_pooling_type='MaxPool1d', scalar_head_fc_layers=1,
               ocr_start=375, ocr_end=625, output_dir='output', model_structure=None,
               seq_len=998, get_complete_corr_metrics=False,
               save_file_name=None): # get_complete_corr_metrics means it will also get corr across peaks and spearman corr across celltypes
    
    metric_names = [
        'scalar_corr', 'profile_corr', 'jsd', 'ocr_profile_corr',
        'ocr_jsd', 'top_ocr_profile_corr', 'top_ocr_jsd'
    ]

    if get_complete_corr_metrics:
        metric_names = [
            'scalar_corr', 'profile_corr', 'jsd', 'ocr_profile_corr',
            'ocr_jsd', 'top_ocr_profile_corr', 'top_ocr_jsd', 
            'scalar_pearson_corr_across_ocrs', 'scalar_spearman_corr_x_celltypes', 'scalar_spearman_corr_x_ocrs'
        ]
    print('BIN SIZE IS', bin_size)

    metric_values = model_analysis_from_saved_model(
        saved_model_path=saved_model_path,
        n_celltypes=n_celltypes,
        n_filters=n_filters,
        infofile_path=infofile_path,
        get_scalar_corr=get_scalar_corr,
        get_profile_corr=get_profile_corr,
        get_jsd=get_jsd,
        eval_set=eval_set,
        bin_size=bin_size,
        bin_pooling_type=bin_pooling_type,
        scalar_head_fc_layers=scalar_head_fc_layers,
        ocr_start=ocr_start,
        ocr_end=ocr_end,
        model_structure=model_structure,
        seq_len=seq_len,
        get_complete_corr_metrics=get_complete_corr_metrics,
    )
    metrics = dict(zip(metric_names, metric_values))

    save_metrics(metrics, output_dir, eval_set, save_file_name)
    print("shape of top ocr corr and jsd", metrics['top_ocr_profile_corr'].shape, metrics['top_ocr_jsd'].shape)
    plot_top_ocr_profile_corr(metrics['top_ocr_profile_corr'], output_dir, eval_set)
    plot_top_jsd(metrics['top_ocr_jsd'], output_dir, eval_set)
    plot_scalar_corr(metrics['scalar_corr'], output_dir, eval_set)

def get_model_structure(model_type:str, n_filters, n_celltypes, bin_size=1, seq_len=998, off_by_two=True):
    if model_type == "BPcm":
        return BPcm(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, scalar_head_fc_layers=1)
    if model_type == "BPcm_250":
        return BPcm_250(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, scalar_head_fc_layers=1, off_by_two=off_by_two)
    elif model_type == "BPcm_bias0":
        return BPcm_bias0(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, scalar_head_fc_layers=1)
    elif model_type == "BPnetRep":
        return BPnetRep(seq_len=seq_len, n_celltypes=n_celltypes, num_filters=n_filters)
    elif model_type == 'BPol':
        return BPol(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size)
    elif model_type == 'BPmp':
        return BPmp(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size)
    elif model_type == 'BPcm_skinny':
        return BPcm_skinny(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size)
    elif model_type == 'BPcm_super_skinny':
        return BPcm_super_skinny(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size)
    elif model_type == 'BPbi':
        return BPbi(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, scalar_head_fc_layers=1)
    elif model_type == 'BPbi_shallow':
        return BPbi_shallow(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, scalar_head_fc_layers=1)
    else:
        raise Exception("The provided model type is not supported")

def main(): # Use main method to access eval_model through the command line
    args = parse_arguments()
    print(args)

    model_structure = get_model_structure(args.model_type, args.n_filters, args.n_celltypes, bin_size=args.bin_size, seq_len=args.seq_len, off_by_two=args.off_by_two)
    print("eval model model structure", model_structure)


    eval_model(
        saved_model_path=args.saved_model_path,
        n_celltypes=args.n_celltypes,
        n_filters=args.n_filters,
        infofile_path=args.infofile_path,
        get_scalar_corr=args.get_scalar_corr,
        get_profile_corr=args.get_profile_corr,
        get_jsd=args.get_jsd,
        eval_set=args.eval_set,
        bin_size=args.bin_size,
        bin_pooling_type=args.bin_pooling_type,
        scalar_head_fc_layers=args.scalar_head_fc_layers,
        ocr_start=args.ocr_start,
        ocr_end=args.ocr_end,
        output_dir=args.output_dir,
        model_structure=model_structure,
        seq_len=args.seq_len,
        get_complete_corr_metrics=args.get_complete_corr_metrics,
        save_file_name=args.save_file_name
    )



if __name__ == '__main__':
    main()