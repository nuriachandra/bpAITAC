import numpy as np
from utils.load_model import get_scalar_correlation, model_analysis, load_data
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from matplotlib.colors import ListedColormap
from plotting.combine_results import find_files, combine_trial_correlations, combine_trial_metrics, combine_multiple_models, N_PEAKS_PER_SET
from torch import nn
from plot_utils_bpaitac import get_coefficient_of_variance_quartile_array

def add_evenly_spread_points(ax, data, x_col, y_col, width=0.4, color='black', alpha=0.6, size=20):
    """
    Adds evenly spread points to an existing boxplot.
    
    :param ax: The matplotlib axis object of the existing plot
    :param data: DataFrame containing the data
    :param x_col: Name of the column for x-axis categories
    :param y_col: Name of the column for y-axis values
    :param width: Width of the spread within each category
    :param color: Color of the points
    :param alpha: Transparency of the points
    :param size: Size of the points
    """
    grouped = data.groupby(x_col)
    
    for idx, (name, group) in enumerate(grouped):
        y_values = group[y_col].values
        n = len(y_values)
        
        # Calculate evenly spaced x positions
        if n == 1:
            x_spread = [0]
        else:
            x_spread = np.linspace(-width/2, width/2, n)
        
        # Plot the points
        ax.scatter(x_spread + idx, y_values, color=color, alpha=alpha, s=size, zorder=3)

def plot_boxplot(data, trial_df, out_path, x_label, y_label, title="", 
                 mean_y_label=None, figure_ratio=(11.7, 8.27), fliers=True, plot_area_size=None): # if none is given defaults to same as y_label
    """
    plots boxplot of the data
    Also plots in a scatter plot the mean of the trials on top
    """
    print("arguments to plot boxplot:")
    print("data", data)
    print("trial df", trial_df)
    print("outpath", out_path)
    print("x label", x_label)
    print("y label", y_label)
    print("title", title)
    print("datatypes", data.dtypes)

    if x_label == 'lambda':
        axis_x_label = "$\lambda$"
    else:
        axis_x_label = x_label

    axis_y_label = y_label
    if y_label == "scalar_corr":
        axis_y_label = "Pearson R of OCRs"
    elif y_label == "scalar_corr_peaks":
        axis_y_label = "Pearson R of Celltypes"
    elif y_label == "scalar_spearman_corr_x_celltypes":
        axis_y_label = "Spearman correlation across cells"
    elif y_label == "bp_corr_mean":
        axis_y_label = "Mean Pearson R of Profiles"
    elif y_label == "scalar_pearson_corr_across_ocrs":
        axis_y_label = "Pearson R of Celltypes"
    if mean_y_label is None:
        mean_y_label = y_label

    plt.clf()
    plt.figure(figsize=figure_ratio)

    sns.set(rc={'figure.figsize':figure_ratio})

    # Create a rainbow color palette
    # palette = sns.color_palette("rainbow", n_colors=len(data[x_label].unique()))
    palette = sns.color_palette("Blues", n_colors=len(data[x_label].unique()))

    # Use the rainbow color palette for boxplot
    graph = sns.boxplot(x=x_label, y=y_label, data=data, palette=palette, showfliers=fliers)
    graph.set_xticklabels(graph.get_xticklabels(), rotation=40, ha="right")

    plt.title(title)

    print('x labels', trial_df[x_label], 'mean_y_label', mean_y_label, trial_df[mean_y_label])
    # adding trial data points
    # sns.stripplot(x=x_label, y=mean_y_label, data=trial_df, color='k', size=8, jitter=0.3)
    add_evenly_spread_points(plt.gca(), trial_df, x_label, mean_y_label)

    # add_evenly_spread_points(plt.gca(), data, trial_df[x_label], mean_y_label)

    graph.set_xlabel(axis_x_label, fontsize=10)
    graph.set_ylabel(axis_y_label, fontsize=10)
    graph.tick_params(labelsize=10)

    # If plot_area_size is specified, adjust the plot area
    if plot_area_size:
        # Check if values are absolute (greater than 1) or proportional
        if plot_area_size[0] > 1 or plot_area_size[1] > 1:
            # Convert absolute sizes (inches) to proportions
            width_prop = plot_area_size[0] / figure_ratio[0]
            height_prop = plot_area_size[1] / figure_ratio[1]
            # Ensure proportions don't exceed 1
            width_prop = min(width_prop, 0.95)
            height_prop = min(height_prop, 0.95)

        right_margin = 0.03
        x_margin = 1 - right_margin - width_prop
        y_margin = 0.97 - height_prop
        graph.set_position([x_margin, y_margin, width_prop, height_prop])


    # Save the plot
    plt.savefig(out_path + '.png', bbox_inches='tight', dpi=330)
    print('saved fig at', out_path + '.png')
    plt.show()
    plt.clf()


def plot_boxplot_cv_quartile(data, trial_df, out_path, x_label, y_label, info_path, title="", 
                 mean_y_label=None, figure_ratio=(3.5, 4.5)):
    """
    Plots boxplot of the data, with quartiles on x-axis and boxes colored by x_label.
    Also plots in a scatter plot the mean of the trials on top.
    """
    plt.clf()

    cv_quartile_arr = get_coefficient_of_variance_quartile_array(info_path)
    assert len(data[y_label]) % len(cv_quartile_arr) == 0
    n_trials_total = len(data[y_label])//len(cv_quartile_arr)
    cv_quartile = np.tile(cv_quartile_arr, n_trials_total)
    data['cv_quartile'] = cv_quartile

    axis_y_label = y_label
    if y_label == "scalar_corr":
        axis_y_label = "Pearson R of OCRs"
    elif y_label == "scalar_corr_peaks":
        axis_y_label = "Pearson R across regions"
    if mean_y_label is None:
        mean_y_label = y_label

    plt.clf()
    fig, ax = plt.subplots(figsize=figure_ratio)
    sns.set(style="whitegrid")

    # Create a color palette for x_label
    x_label_palette = sns.color_palette("husl", n_colors=len(data[x_label].unique()))

    # Create the boxplot
    sns.boxplot(x='cv_quartile', y=y_label, hue=x_label, data=data, palette=x_label_palette, ax=ax)

    # Remove x-axis labels
    ax.set_xticklabels([1, 2, 3, 4])
    ax.set_xlabel("Quartiles CV of OCRs", fontsize=10)
    
    ax.set_title(title, fontsize=10)

    # Adding trial data points
    if 'trial' in data.columns:
        # Group by cv_quartile and trial, then calculate mean
        trial_means = data.groupby(['cv_quartile', 'trial', x_label])[y_label].mean().reset_index()
        
        for i, (name, group) in enumerate(trial_means.groupby(x_label)):
            # Calculate x-positions for the scatter points based on the group index
            if i == 0:
                x_positions = group['cv_quartile'].astype(float) - 1 - 1/5
            elif i == 1:
                x_positions = group['cv_quartile'].astype(float) - 1 + 1/5
            else:
                print("error: only two groups supported right now")
                exit()
            # x_jitter = np.random.normal(0, 0.05, len(x_positions))
            ax.scatter(x_positions, group[y_label], color='black',
                    s=20, alpha=0.7, zorder=3)
    


    ax.set_ylabel(axis_y_label, fontsize=10)

    # Adjust legend
    # handles, labels = ax.get_legend_handles_labels()
    # unique_labels = dict(zip(labels, handles))
    # ax.legend(unique_labels.values(), unique_labels.keys(), 
    #           title=x_label, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust legend
    # if x_label == 'lambda':
    #     legend_title = "$\lambda$"
    # else:
    #     legend_title = x_label
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(out_path + '.png', bbox_inches='tight', dpi=330)
    print('saved boxplot at', out_path + '.png')
    plt.show()
    plt.clf()


def top_OCRs(total_counts:np.ndarray, n_celltypes:int, percent:int=10):
    """
    returns numpy array of the indices of the OCRs of the top P percent
        of expressed OCRs for all celltypes 

    total_counts should be for the validation data if you are 
        planning on analyzing validation data 
    """
    # get total counts for the desired celltype 
    print("total num ocrs:", len(total_counts))

    # Calculate how many 10% of the elements is
    num_top = int(len(total_counts)*0.1)
    print(num_top)
    indices = np.zeros(shape=(num_top, n_celltypes), dtype=int)
    print("indices shape", indices.shape)
    for i in range(n_celltypes):
        cell_counts = total_counts[:, i]
        sorted_indices = np.argsort(cell_counts)
        top_p_indices = np.argsort(cell_counts)[-num_top:]
        # top_ocrs = cell_counts[top_p_indices]
        # indices[: i] = top_ocrs
        print("for celltype", i, "top ocr indices are", top_p_indices[-4:])
        indices[:, i] = top_p_indices
    return indices 

def get_cell_metrics(val_total_counts:np.ndarray, celltypes:np.ndarray, ocr_names:np.ndarray, results_dir, lambda_dirs, lambdas):
    """
    gets the metrics top 10 percent of expressed OCRs in each cell
    (note the ocrs are different for each celltype)

    returns a list containing dataframes with metrics for each celltype
    the order of celltypes corresponds to the input data which in theory
    is the same as the celltypes array
    """
    n_celltypes = len(celltypes)
    print("number of celltypes", n_celltypes)
    # TODO CHANGE DIM TO BE LOADED FOR SAMPLE AND COMPLETE
    
    top_idx = top_OCRs(val_total_counts, n_celltypes, percent=10)
    print("The top 10percent of OCRs is n=", len(top_idx))
    # for each celltype:

    # make list of dfs - one for each celltype
    # do we want to make pandas df for each celltype? I think yes bc easy to plot
    column_names = ['ocr names', 'bp_corr', 'jsd', 'lambda', 'trial']
    df_list = [pd.DataFrame(columns=column_names) for i in range(n_celltypes)]
    # trial_df_list = [pd.DataFrame(columns=column_names) for i in range(n_celltypes)]

    trials_df_list = [pd.DataFrame(columns=column_names) for i in range(n_celltypes)]
    # For every lambda:
    for i in range(len(lambda_dirs)):
        lambda_dir = lambda_dirs[i]
        lambda_ = lambdas[i]
        cur_dir = os.path.join(results_dir, lambda_dir)
        print("current dir:", lambda_dir)
        analysis_files = find_files(cur_dir, 'analysis.npz')
        print("founds paths:", analysis_files)
        
        # For every trial: 
        for k, file in enumerate(analysis_files):
            analysis = np.load(file)
            bp_corrs, jsd = analysis['bp_corr'], analysis['jsd']
            # Record the metrics for top p OCRs for each celltype
            for j in range(n_celltypes):
                ocr_idx = top_idx[:, j]
                # trial_df = trial_df_list[j]
                # add each type of metric to this

                # make a df just for this trial and celltype, and append to df for previous trials of this cell
        
                print("jsd selected", jsd[ocr_idx, j].shape, pd.Series(jsd[ocr_idx, j])) # correct jsd is selected 
                curr_analysis_df = pd.DataFrame({
                                'ocr names':ocr_names[ocr_idx],
                                'bp_corr':pd.Series(bp_corrs[ocr_idx, j]),
                                'jsd':pd.Series(jsd[ocr_idx, j]),
                                'lambda':np.full(len(ocr_idx), lambda_, dtype=float),
                                'trial':np.full(len(ocr_idx), k+1)})

                df_list[j] = pd.concat([df_list[j], curr_analysis_df], axis=0, ignore_index=True)
                
                trial_mean_row = {'bp_corr': np.mean(bp_corrs[ocr_idx, j]), 'jsd': np.mean(jsd[ocr_idx, j]), 'lambda': lambda_, 'trial': k+1}
                trials_df_list[j] = trials_df_list[j].append(trial_mean_row, ignore_index=True)

    return df_list, trials_df_list
    

# # MAIN

# # FOR COMPLETE
# n_celltypes = 90
# # in chelan
# val_info_path = '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt'
# # in hyak 
# # val_info_path = '/gscratch/mostafavilab/nchand/data/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_4.3.23-2/memmap/info.txt'
# # results_dir = '/gscratch/mostafavilab/nchand/results/analysis' # the directory where all the lambda files are housed in 

# # default bin size and pooling etc
# bin_size= 1
# bin_pooling_type = nn.MaxPool1d
# scalar_head_fc_layers = 1

# #TODO for lambda based experiments you must set experiment names to the list of lambdas
# x_axis_var = 'lambda'


# For BP6
# model_name = 'BPnetRep'
# results_dir = '/data/nchand/analysis/' # the directory where all the lambda files are housed in 
# lambda_dirs = ['BP6_LZ', 'BP6_L-17', 'BP6_L-15', 'BP6_L-13', 'BP6_L-12', 'BP6_L-11', 'BP6_L-9', 'BP6_L-7', 'BP6_L-5', 'BP6_L-3', 'BP6_L-1']
# lambdas = [0, 10**(-17), 10**(-15), 10**(-13), 10**(-12), 10**(-11), 10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1)]# list of the corresponding lambda values in scientific notation
# n_filters = 64

# For BP13
# results_dir = '/data/nchand/analysis/BPnetRep/'
# lambda_dirs = ['BP13_LZ', 'BP13_L-10', 'BP13_L-4', 'BP13_L-2', 'BP13_L-1', 'BP13_L0_1', 'BP13_L1', 'BP13_L2', 'BP13_L4']
# lambdas = [0, 10**(-10), 10**(-4), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(4)]# list of the corresponding lambda values in scientific notation
# n_filters = 64

# for BPcm 
# model_name = 'BPcm'
# model_number = 'BP17'
# boxplot_iteration_number = '8T'
# results_dir = '/data/nchand/analysis/BPcm/' # the directory where all the lambda files are housed in 
# n_filters = 300

# for BP16
# lambda_dirs = ['BP16_LZ', 'BP16_L-10', 'BP16_L-6', 'BP16_L-2', 'BP16_L0_1', 'BP16_L6']
# lambdas = [0, 10**(-10), 10**(-6), 10**(-2), 10**(0), 10**(6)]# list of the corresponding lambda values in scientific notation

# for BP17 complete
# lambda_dirs = ['BP17_L0_0', 'BP17_L-2_1', 'BP17_L-1_1', 'BP17_L-1_2', 'BP17_L-1_3', 'BP17_L-1_4', 'BP17_L-1_5', 'BP17_L-1_6', 'BP17_L-1_7', 'BP17_L-1_8', 'BP17_L-1_9',
#               'BP17_L0_1', 'BP17_L0_1.1', 'BP17_L0_1.2', 'BP17_L0_1.3', 'BP17_L0_1.4', 'BP17_L0_1.5', 'BP17_L0_1.6', 'BP17_L0_1.7', 'BP17_L0_1.8', 'BP17_L0_1.9', 'BP17_L0_2',
#               'BP17_L2_1', 'BP17_L4_1', 'BP16_L10']
# lambdas = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#            1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
#            100, 10000, 10000000000]
# lambda_dirs = ['BP17_L0_0', 'BP17_L-2_1', 'BP17_L-1_1', 'BP17_L-1_3', 'BP17_L-1_7', 'BP17_L-1_8', 'BP17_L-1_9',
#               'BP17_L0_1', 'BP17_L0_1.1', 'BP17_L0_1.3', 'BP17_L0_1.7', 'BP17_L0_2',
#               'BP17_L2_1', 'BP17_L4_1', 'BP16_L10']
# lambdas = [0, 0.01, 0.1, 0.3, 0.7, 0.8, 0.9,
#            1, 1.1, 1.3, 1.7, 2,
#            100, 10000, 10000000000]
# lambda_dirs = ['BP17_L0_0', 'BP17_L-1_1', 'BP17_L-1_3', 'BP17_L-1_5', 'BP17_L-1_7', 'BP17_L-1_9',
#               'BP17_L0_1.1', 'BP17_L0_1.3', 'BP17_L0_1.5']
# lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9,
#              1.1, 1.3, 1.5]



# BP18
# lambda_dirs = ['BP18_L0_0', 'BP18_L-2_1', 'BP18_L-1_1', 'BP18_L-1_7', 'BP18_L-1_8', 'BP18_L-1_9', 
#                'BP18_L0_1', 'BP18_L0_1.1', 'BP18_L0_1.2', 'BP18_L0_2', 'BP18_L2_1', 'BP18_L6_1', 'BP18_L10_1']
# lambdas = [0, 0.01, 0.1, 0.7, 0.8, 0.9,
#            1, 1.1, 1.2, 2, 10**2, 10**6, 10**10]

# BP21
# lambda_dirs = ['BP21_L0_1', 'BP21_L2_1', 'BP21_L3_1']
# lambdas = [1, 100, 1000]

# BP22
# model_name = 'BPcm'
# model_number = 'BP22'
# eval_set = 'training'
# boxplot_iteration_number = 'training'
# results_dir = '/data/nchand/analysis/BPcm/' # the directory where all the lambda files are housed in 
# n_filters = 300
# lambda_dirs = ['BP22_L0_0', 'BP22_L0_0.5', 'BP22_L0_0.75', 'BP22_L0_1']
# lambdas = [0, 0.5, 0.75, 1]

# BP27
# for the lambda run:
# model_name = 'BPcm'
# model_number = 'BP27'
# eval_set = 'validation'
# boxplot_iteration_number = str(4)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# base_dir = 'BP27_1_L0'
# lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# lambda_dirs = [f"{base_dir}_{value:.1f}" if value % 1 != 0 else f"{base_dir}_{int(value)}" for value in lambdas]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='analysis2.npz'


# BP27
# for the lambda run:
# model_name = 'BPcm'
# model_number = 'BP27'
# eval_set = 'validation'
# boxplot_iteration_number = str(4)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# base_dir = 'BP27_1_L0'
# lambdas = [0, 0.1]
# lambda_dirs = [f"{base_dir}_{value:.1f}" if value % 1 != 0 else f"{base_dir}_{int(value)}" for value in lambdas]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='analysis2.npz'

# BP27
#for the diff bin sizes run:
# model_name = 'BPcm'
# model_number = 'BP27'
# eval_set = 'validation'
# boxplot_iteration_number = str(11)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# base_dir = 'BP27_1_L0'
# lambda_dirs = ['BP27_1_L0_0.5', 'BP27_3_L0_0.5', 'BP27_5_L0_0.5', 'BP27_20_L0_0.5']
# lambdas = [0.5] * len(lambda_dirs)
# bin_sizes = [1, 3, 5, 20]
# dirs=lambda_dirs
# model_names = lambda_dirs
# experiment_names=bin_sizes
# x_axis_var='experiment_name'

# BP28
# model_name = 'BPcm'
# model_number = 'BP28'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPcm/tooBig/' # it's on hyak
# n_filters = 300
# lambdas = [0.9, 0.9, 0.9, 0.9, 0.9]
# dirs = ['BP28_1_L0_0.9', 'BP28_2_L0_0.9', 'BP28_3_L0_0.9', 'BP28_5_L0_0.9', 'BP28_10_L0_0.9']
# bin_sizes = [1, 2, 3, 5, 10]

# experiment_names = bin_sizes
# bin_size= 1
# bin_pooling_type = nn.MaxPool1d
# scalar_head_fc_layers = 1
# x_axis_var = 'experiment_name'

# BP29
# model_name = 'BPcm'
# model_number = 'BP29'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPcm/tooBig/'
# n_filters = 1000
# dirs = ['BP29_1_L0_0.9', 'BP29_2_L0_0.9', 'BP29_3_L0_0.9', 'BP29_5_L0_0.9', 'BP29_10_L0_0.9', 'BP29_20_L0_0.9']
# bin_sizes = [1, 2, 3, 5, 10, 20]
# lambdas = [0.5] * len(dirs)
# experiment_names = bin_sizes
# bin_size= 1
# bin_pooling_type = nn.MaxPool1d
# scalar_head_fc_layers = 1
# x_axis_var = 'experiment_name'

# BP33
# NOTE MUST BE IN APPROPRIATE GIT BRANCH bin-at-end
# model_name = 'BPcm'
# model_number = 'BP33'
# eval_set = 'validation'
# boxplot_iteration_number = str(2)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# dirs = ['BP33_1_L0_0.9', 'BP33_2_L0_0.9', 'BP33_3_L0_0.9', 'BP33_5_L0_0.9', 'BP33_10_L0_0.9', 'BP33_20_L0_0.9']
# # dirs = ['BP33_1_L0_0.9', 'BP33_5_L0_0.9', 'BP33_10_L0_0.9', 'BP33_20_L0_0.9']
# bin_sizes = [1, 2, 3, 5, 10, 20]
# lambdas = [0.9] * len(dirs)
# experiment_names = bin_sizes
# bin_pooling_type = 'none'
# scalar_head_fc_layers = 1
# x_axis_var = 'experiment_name'

# BP56
# model_name = 'BPcm'
# model_number = 'BP56'
# eval_set = 'validation'
# boxplot_iteration_number = str(2)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# base_dir = 'BP56_20_L'
# lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# lambda_dirs = [
#     "BP56_20_L0_0",
#     "BP56_20_L-1_1",
#     "BP56_20_L-1_2",
#     "BP56_20_L-1_3",
#     "BP56_20_L-1_4",
#     "BP56_20_L-1_5",
#     "BP56_20_L-1_6",
#     "BP56_20_L-1_7",
#     "BP56_20_L-1_8",
#     "BP56_20_L0_1"
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='analysis.npz'

# BP60
# model_name = 'BPcm'
# model_number = 'BP60'
# eval_set = 'validation'
# boxplot_iteration_number = str(2)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# base_dir = ''
# lambdas = [0.0, 10**-3, 10**-2, 8*10**-2, 0.4, 0.5, 0.6, 0.7]
# lambda_dirs = [
#     "BP60_20_L0_0",
#     "BP60_20_L-3_1",
#     "BP60_20_L-2_1",
#     "BP60_20_L-2_8",
#     "BP60_20_L-1_4",
#     "BP60_20_L-1_5",
#     "BP60_20_L-1_6",
#     "BP60_20_L-1_7",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='analysis.npz'


# for the all celltype 50000 OCR sample data
# celltypes='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/cell_names.npy'
# val_info_path='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/info.txt'

# complete data in Chelan
# val_total_counts = np.memmap('/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/val.total_counts.dat', 
#                              dtype='float32', shape=(28329, 90))
# ocr_names = np.memmap('/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/val.names.dat', dtype='<U26', shape=(28329))
# celltypes = np.load("/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy")

# BP65
# model_name = 'BPcm'
# model_number = 'BP65'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# lambda_dirs = [
#     "BP65_L0_0",
#     "BP65_L-1_1",
#     "BP65_L-1_2",
#     "BP65_L-1_3",
#     "BP65_L-1_4",
#     "BP65_L-1_5",
#     "BP65_L-1_6",
#     "BP65_L-1_7",
#     "BP65_L-1_8",
#     "BP65_L-1_9",
#     "BP65_L0_1",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'

# BP66
# model_name = 'BPcm'
# model_number = 'BP66'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]

# lambda_dirs = [
#     "BP66_L-1_1",
#     "BP66_L-1_2",
#     "BP66_L-1_3",
#     "BP66_L-1_4",
#     "BP66_L-1_5",
#     "BP66_L-1_6",
#     "BP66_L-1_7",
#     "BP66_L-1_9",
#     "BP66_L0_1",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'

# BP68
# model_name = 'BPcm'
# model_number = 'BP68'
# eval_set = 'validation'
# boxplot_iteration_number = str(2)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]

# lambda_dirs = [
#     "BP68_L0_0",
#     "BP68_L-1_1",
#     "BP68_L-1_2",
#     "BP68_L-1_3",
#     "BP68_L-1_4",
#     "BP68_L-1_5",
#     "BP68_L-1_6",
#     "BP68_L-1_7",
#     "BP68_L-1_9",
#     "BP68_L0_1",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'

# # BP68
# model_name = 'BPcm'
# model_number = 'BP68'
# eval_set = 'testing'
# boxplot_iteration_number = "t_1"
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# lambdas = [0, 0.5]

# lambda_dirs = [
#     "BP68_L0_0",
#     "BP68_L-1_5",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='testing_analysis.npz'

# BP69
# model_name = 'BPcm'
# model_number = 'BP69'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPcm/'
# n_filters = 300
# lambdas = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9, 1.0]

# lambda_dirs = [
#     "BP69_L0_0",
#     "BP69_L-1_1",
#     "BP69_L-1_1.5",
#     "BP69_L-1_2",
#     "BP69_L-1_2.5",
#     "BP69_L-1_3",
#     "BP69_L-1_5",
#     "BP69_L-1_7",
#     "BP69_L-1_9",
#     "BP69_L0_1",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'

# BP70
# model_name = 'BPnetRep'
# model_number = 'BP70'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPnetRep/'
# n_filters = 300
# lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]

# lambda_dirs = [
#     "BP70_L0_0",
#     "BP70_L-1_1",
#     "BP70_L-1_2",
#     "BP70_L-1_3",
#     "BP70_L-1_4",
#     "BP70_L-1_5",
#     "BP70_L-1_7",
#     "BP70_L-1_8",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'


# BP71
# model_name = 'BPnetRep'
# model_number = 'BP71'
# eval_set = 'validation'
# boxplot_iteration_number = str(2)
# results_dir = '/data/nchand/analysis/BPnetRep/'
# n_filters = 300
# lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]

# lambda_dirs = [
#     "BP71_L0_0",
#     "BP71_L-1_1",
#     "BP71_L-1_2",
#     "BP71_L-1_3",
#     "BP71_L-1_4",
#     "BP71_L-1_5",
#     "BP71_L-1_7",
#     "BP71_L-1_9",
#     "BP71_L0_1",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'

# BP73
# model_name = 'BPnetRep'
# model_number = 'BP73'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPnetRep/'
# n_filters = 300
# lambdas = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]

# lambda_dirs = [
#     "BP73_L0_0",
#     "BP73_L-1_1",
#     "BP73_L-1_2",
#     "BP73_L-1_25",
#     "BP73_L-1_3",
#     "BP73_L-1_4",
#     "BP73_L-1_5",
#     "BP73_L-1_7",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'

# BP74
# model_name = 'BPnetRep'
# model_number = 'BP74'
# eval_set = 'validation'
# boxplot_iteration_number = str(1)
# results_dir = '/data/nchand/analysis/BPnetRep/'
# n_filters = 300
# lambdas = [0.0, 0.3, 0.5, 0.9]

# lambda_dirs = [
#     "BP74_L0_0",
#     "BP74_L-1_3",
#     "BP74_L-1_5",
#     "BP74_L-1_9",
# ]
# bin_sizes = [1] * len(lambda_dirs)
# dirs=lambda_dirs
# print(lambda_dirs)
# model_names = lambda_dirs
# experiment_names=lambdas
# x_axis_var='lambda'
# analysis_file_name='validation_analysis.npz'



# for the all celltype 50000 OCR sample data
# celltypes='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/cell_names.npy'
# val_info_path='/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_random5000_allcell_bias_corrected_normalized_5.28.23/memmap/info.txt'

# complete data in Chelan
# val_total_counts = np.memmap('/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/val.total_counts.dat', 
#                              dtype='float32', shape=(28329, 90))
# ocr_names = np.memmap('/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/val.names.dat', dtype='<U26', shape=(28329))
# celltypes = np.load("/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy")


# n_filters = 300
# lambda_dirs = ['BP10_L-12', 'BP10_L-5']
# lambdas = [10**(-12), 10**(-5)]

# for sample
# n_celltypes = 6
# n_filters = 64
# val_info_path = '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_bias_corrected_normalized_3.7.23/memmap/info.txt'
# results_dir = '/homes/gws/nchand/MostafaviLab/results/BPnetRep/'
# lambda_dirs = ['BP6_L-1_1/sample', 'BP6_L-20/sample'] # list of the directory names
# lambdas = [10**(-1), 10**(-20)]
# model_names=lambda_dirs

# analysis_df, trial_df = combine_multiple_models(results_dir, dirs, lambdas, val_info_path, n_celltypes, n_filters, bin_sizes, bin_pooling_type, scalar_head_fc_layers, eval_set=eval_set, experiment_names=experiment_names, analysis_file_name=analysis_file_name)
# # make a lambda_label column that has the lambdas in scientific notation
# analysis_df['sci_lambda'] = analysis_df['lambda'].apply(lambda x: "{:.2e}".format(x))
# trial_df['sci_lambda'] = trial_df['lambda'].apply(lambda x: "{:.2e}".format(x))
# print(analysis_df['sci_lambda'])

# # top_OCRs(total_counts=val_total_counts, n_celltypes=n_celltypes)

# plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_bp_corr_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'bp_corr_mean')
# plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_jsd_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'jsd_mean')
# plot_boxplot(analysis_df,  trial_df, results_dir +  model_number + '_scalar_corr_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'scalar_corr')
# plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_ocr_jsd_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'ocr_jsd_mean')
# plot_boxplot(analysis_df,  trial_df, results_dir +  model_number + '_ocr_bp_corr_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'ocr_bp_corr_mean')
# plot_boxplot(analysis_df,  trial_df, results_dir +  model_number + '_top_ocr_bp_corr_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'top_ocr_bp_corr', mean_y_label='top_ocr_bp_corr_mean')
# plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_top_ocr_jsd_boxplot_' + boxplot_iteration_number, 
#              x_axis_var, 'top_ocr_jsd', mean_y_label='top_ocr_jsd_mean')
# print("output location:", results_dir)

def label_trial_groups(group, n):
    return pd.Series((group.index - group.index[0]) // n + 1, index=group.index)

def main():
    # MAIN
    # FOR COMPLETE
    n_celltypes = 90
    # in chelan
    # val_info_path = '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_shallow_deprotinated_bias_quantile_normalized_4.1.25/memmap/info.txt'
    # in hyak 
    # val_info_path = '/gscratch/mostafavilab/nchand/data/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_4.3.23-2/memmap/info.txt'
    # results_dir = '/gscratch/mostafavilab/nchand/results/analysis' # the directory where all the lambda files are housed in 

    # default bin size and pooling etc
    bin_size= 1
    bin_pooling_type = nn.MaxPool1d
    scalar_head_fc_layers = 1

    #TODO for lambda based experiments you must set experiment names to the list of lambdas
    x_axis_var = 'lambda'

    # Configuration
    model_name = 'BPcm_250'
    model_number = 'BP203'
    eval_set = 'testing'
    boxplot_iteration_number = "v5"
    results_dir = '/data/nchand/analysis/BPcm_250/'
    n_filters = 300
    lambdas = [0, 0.5]
    # lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_celltypes = 90
    val_info_path = '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_shallow_deprotinated_bias_quantile_normalized_4.1.25/memmap/info.txt'

    lambda_dirs = [
        "BP200_L0_0",
        "BP203_no_seed_L-1_5",
    ]

    # lambda_dirs = [
    #     "BP68_L0_0",
    #     "BP68_L-1_1",
    #     "BP68_L-1_2",
    #     "BP68_L-1_3",
    #     "BP68_L-1_4",
    #     "BP68_L-1_5",
    #     "BP68_L-1_6",
    #     "BP68_L-1_7",
    #     "BP68_L-1_8",
    #     "BP68_L-1_9",
    #     "BP68_L0_1",
    # ]

    save_trials=False
    complete_corr_metrics=False

    bin_sizes = [1] * len(lambda_dirs)
    dirs = lambda_dirs
    print(lambda_dirs)
    model_names = lambda_dirs
    experiment_names = lambdas
    x_axis_var = 'lambda'
    analysis_file_name = 'testing_analysis.npz'
    # analysis_file_name = 'testing_correlation_analysis.npz'
    celltype_corr_metrics = False
    max_n_trials = 5 # or none
    out_dir = 'final_figs/'

    # Default settings
    bin_size = 1
    bin_pooling_type = nn.MaxPool1d
    scalar_head_fc_layers = 1

    print('dirs are', dirs)
    # Perform analysis
    analysis_df, trial_df = combine_multiple_models(
        results_dir, dirs, lambdas, val_info_path, n_celltypes, n_filters,
        bin_sizes, bin_pooling_type, scalar_head_fc_layers, eval_set=eval_set,
        experiment_names=experiment_names, analysis_file_name=analysis_file_name,
        max_n_trials=max_n_trials, save_trials=save_trials, complete_corr_metrics=complete_corr_metrics, 
        celltype_corr_metrics=celltype_corr_metrics
    )

    # Add scientific notation for lambda values
    analysis_df['sci_lambda'] = analysis_df['lambda'].apply(lambda x: "{:.2e}".format(x))
    trial_df['sci_lambda'] = trial_df['lambda'].apply(lambda x: "{:.2e}".format(x))

    # I have a dataframe with a column called 'l'
    # for each value of l, calculate the number of entries with that value of l

    if celltype_corr_metrics:
        n=n_celltypes
    else:
        n = N_PEAKS_PER_SET[eval_set]  # Define your group size here
        print('n is ', n)

    # determine how many different values of lambda there are 
    # what if every n just gets a different trial number and we call it good - it would mean like 20 trials
    assert len(analysis_df) % n == 0
    n_trials = len(analysis_df) // n

    trials_arr = np.repeat(np.arange(n_trials), n)
    print('len analysis df', len(analysis_df))
    # print(analysis_df)
    print('len trial df', len(trial_df))
    # print(trial_df)
    analysis_df['trial'] = trials_arr


    print(analysis_df[['lambda', 'trial']])
    analysis_df['model_display_name'] = analysis_df['lambda'].map({0.5: 'bpAI-TAC', 0: 'AI-TAC'})

    plot_boxplot_cv_quartile(analysis_df,  trial_df, out_dir + "1d_"+ model_number + '_scalar_corr_boxplot_' + boxplot_iteration_number, 
                'model_display_name', 'scalar_corr', figure_ratio=(5, 3.5), info_path=val_info_path)
    
        
    # plot_boxplot(analysis_df, trial_df, out_dir + model_number + '_bp_corr_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'bp_corr_mean', figure_ratio=(5.5, 3.5))
    # plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_jsd_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'jsd_mean')
    # plot_boxplot(analysis_df,  trial_df, out_dir +  '2c_4w_' + model_number + '_scalar_corr_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'scalar_corr', figure_ratio=(8,4.75), plot_area_size=(4, 2.75))
    # plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_ocr_jsd_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'ocr_jsd_mean')
    # plot_boxplot(analysis_df,  trial_df, results_dir +  model_number + '_ocr_bp_corr_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'ocr_bp_corr_mean')
    # plot_boxplot(analysis_df,  trial_df, results_dir +  model_number + '_top_ocr_bp_corr_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'top_ocr_bp_corr', mean_y_label='top_ocr_bp_corr_mean')
    # plot_boxplot(analysis_df, trial_df, results_dir + model_number + '_top_ocr_jsd_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'top_ocr_jsd', mean_y_label='top_ocr_jsd_mean')

    # plot_boxplot(analysis_df,  trial_df, out_dir +  "S2c_4w_" + model_number + '_scalar_corr_x_ocr_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'scalar_pearson_corr_across_ocrs', figure_ratio=(7,4.75), fliers=False, plot_area_size=(4, 2.75))
    # plot_boxplot(analysis_df,  trial_df, results_dir +  model_number + '_scalar_spearman_corr_boxplot_' + boxplot_iteration_number, 
    #             x_axis_var, 'scalar_spearman_corr_x_celltypes', figure_ratio=(3.5,9))
    print("output location:", results_dir)

    


if __name__ == "__main__":
    main()


