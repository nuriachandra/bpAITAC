import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
import numpy as np
import os
import sys
import argparse
from BPnetRep import BPnetRep
import CNN_0
from BPme import BPme
from MemmapDataset import MemmapDataset 
import pandas as pd
from functions import JSD, pearson_corr, elementwise_pearson_corr, JSD_numpy, softmax_numpy
import seaborn
from plot_utils import plot_cors
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, pearsonr
from matplotlib.patches import FancyBboxPatch
from eval_model import eval_model
from utils import load_data, get_least_utilized_gpu


from load_model import get_predictions, model_analysis, get_model, load_model
from scipy.stats import gaussian_kde

def plot_training_val_loss(training_file:str, val_file:str, output_file_path:str, num_epochs:int, loss_type:str, model_name: str, title:str):
    """
    plots two different arrays of the same length 
    """
    train_data = np.loadtxt(training_file, delimiter=', ')
    val_data = np.loadtxt(val_file, delimiter=', ')
    plt.plot(train_data[:, 0], train_data[:, 1], label='training')
    plt.plot(val_data[:, 0], val_data[:, 1], label='validation')

    plt.xlabel('Epoch')
    # plt.xticks(np.arange(0, num_epochs + 1, 1))
    plt.ylabel(loss_type)
    plt.title(model_name + " " + title)
    plt.legend()
    plt.savefig((output_file_path+ "/" + title + ".png"))
    plt.close()


def plot_loss(loss_file:str, output_file_path:str, num_epochs:int, title:str):
    """
    plots loss over epochs (could be trianing loss or validation loss)
    
    loss_file: string that represents the path to the txt file that contains 
        the loss per epoch
    output_file_path: the name of the file that the plot will be saved as
    """
    data = np.loadtxt(loss_file, delimiter=', ')
    

    # for i in range(data.shape[0] // num_epochs):
    #     plt.plot(np.arange(num_epochs + 1), data[i*num_epochs : i*num_epochs + num_epochs, 1])
    

    plt.plot(data[:, 0], data[:, 1])

    plt.xlabel('Epoch', fontsize=9)
    plt.xticks(np.arange(0, num_epochs + 1, 1))
    plt.ylabel('Loss', fontsize=18)
    plt.title(title)
    plt.savefig((output_file_path+ "/" + title + ".png"))
    plt.close()

def all_model_correlations():
    """
    Goes through all the files associated with each of the models we
    are interested in and finds the best correlation.
    """
    model_type_path='/homes/gws/nchand/MostafaviLab/results/BPme'
    models = ['M7_L0', 'M7_L2', 'M7_L3', 'M7_L4', 'M7_L10']
    correlations = []
    for model in models:
        dir_path = os.path.join(model_type_path, model)+ '/complete'
        best_corr = 0.0
        print(dir_path)
        for root, dirs, files in os.walk(dir_path, topdown=False): 
            for name in dirs:
                # out of all these runs find the best correlation
                # see if the directory contains complete/
                folder = os.path.join(dir_path, name)
                print(folder)
                corr_file = folder + '/val_correlation.txt'
                if os.path.isfile(corr_file):
                    curr_corr = get_correlations(corr_file)
                    best_corr = max(best_corr, curr_corr)
        correlations += [best_corr]

    print(correlations)
    barplot_correlations(correlations, models, os.path.join(model_type_path, 'M7.png'))

def get_correlations(filepath:str):
    corr=np.loadtxt(filepath, delimiter=', ')
    return np.max(corr[:,1])

def barplot_correlations(correlations, model_names, out_save_path):
    idx = np.arange(len(correlations))
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.set_xticks(idx)
    print(correlations)
    print(len(idx))
    plt.bar(idx, correlations, color='b')
    # plt.ylabel('Pearson Cor. of Predicted and Actual Total Counts in OCR')
    plt.ylabel('Pearson Cor. of Predicted and Actual')
    # plt.xlabel('Filters')
    ax1.set_xticklabels(model_names)
    plt.savefig(out_save_path)
    plt.show()

def plot_specific_correlations():
    names= ['BPNet Lambda=0', 'BPNet Lambda=10^3', 'BPNet Lambda=10^10']
    dirs = ['/homes/gws/nchand/MostafaviLab/results/BPnetRep/T38/complete/12-29-2022.23.01',
            '/homes/gws/nchand/MostafaviLab/results/BPnetRep/T38/complete/01-04-2023.13.29',
            '/homes/gws/nchand/MostafaviLab/results/BPnetRep/T38/complete/01-05-2023.11.17']
    corrs = []
    for folder in dirs:
        path = os.path.join(folder, 'val_correlation.txt')
        print(path)
        corrs+= [get_correlations(path)]
    barplot_correlations(corrs, names, '/homes/gws/nchand/MostafaviLab/results/BPnetRepBarPlt')
    print(corrs)

def plot_averages():
    # FOR AITAC VS BPNET
    # models = ['BP1_L0', 'BP1_L3', 'BP1_L10']
    # names = ['AI-TAC', 'BPNet Lambda=0', 'BPNet Lambda=10^3', 'BPNet Lambda=10^10']
    # model_type_path = '/homes/gws/nchand/MostafaviLab/results/BPnetRep'
    # correlations = [0.325]

    # FOR T46 - modified BPnet
    models = ['T46_200', 'T46_400']
    names = ['AITAC', '200 filters', '400 filters']
    model_type_path = '/homes/gws/nchand/MostafaviLab/results/BPnetRep'
    correlations = [0.325]
    for model in models:
        dir_path = os.path.join(model_type_path, model)+ '/complete'
        n_trials = 0.0
        sum_corr = 0.0
        print(dir_path)
        for root, dirs, files in os.walk(dir_path, topdown=False): 
            for name in dirs:
                # out of all these runs find the best correlation
                # see if the directory contains complete/
                n_trials+=1
                folder = os.path.join(dir_path, name)
                print(folder)
                corr_file = folder + '/val_correlation.txt'
                if os.path.isfile(corr_file):
                    curr_corr = get_correlations(corr_file)
                    sum_corr += curr_corr
        avg = sum_corr / n_trials
        print(avg)
        correlations += [avg]

    print(correlations)
    barplot_correlations(correlations, names, '/homes/gws/nchand/MostafaviLab/results/BPnetRep/T46_correlations')

# def box_plot():
#     hyak_info_file_dir="/gscratch/mostafavilab/nchand/bpAITAC/data_train_test/sample_normalized_250center_1.17.23/memmap/info.txt"
#     celltype_path = "/gscratch/mostafavilab/nchand/data/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy"
#     celltypes_list = np.load(celltype_path)
#     model_path = "/gscratch/mostafavilab/nchand/results/BPnetRep/BP1_L10/complete/01-18-2023.08.17/best_model"
#     model = BPnetRep(1000, celltypes_list)
#     get_all_correlations(model_path, model, hyak_info_file_dir)

def get_all_correlations(model_path:str, model:nn.Module, info_file):
    """
    This method loads a model and makes predictions on the validation set 
    model: is the path to the model which was saved with model.state dict of the
        best found weights
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)  # this should print out CUDA 
    model.to(DEVICE)
    train_loader, val_loader, test_loader = load_data(info_file, batch_size=100)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    profile_predictions = []
    for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(val_loader):
        x = seqs.to(DEVICE)
        y = bp_counts.to(DEVICE)
        bias = torch.squeeze(seq_bias).to(DEVICE) # the squeezing to work with current head setup
        profile, scalar = model(x, bias)
        profile_predictions += Tensor.toList(profile)
    print(profile_predictions)

def get_correlations(prediction:np.ndarray, actual:np.ndarray, out_file):
    corr = elementwise_pearson_corr(prediction, actual)
    print("correlation size ", corr.shape)
    np.savez(out_file, correlations=corr)

def boxplot(data, filepath, title):
    """
    datais expected to be a  np arrays 
    """
    # find correlations
    plt.clf()
    plt.boxplot(data)
    plt.ylabel("pearson correlation")
    plt.xlabel(title)
    plt.title(title)
    plt.savefig(filepath)

def scatter_plot(x:np.ndarray, y:np.ndarray, x_label, y_label, title, filepath):
    """
    expects data to be of the same dimension
    flattens arrays and plots them
    """
    x, y = x.flatten(), y.flatten()
    fig = plt.figure()
    
    colors = gaussian_kde(np.array([x, y]))(np.array([x, y]))
    sort = np.argsort(np.abs(colors))
    plt.scatter(x[sort], y[sort], cmap = 'cividis', c=colors[sort], alpha = 0.5)
    # plt.scatter(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filepath)


def get_JSD(obs, pred):
    """
    obs and pred should be numpy arrays.
    or np memmaps that can be treated as np arrays
    Expected shape is (N, num_cell_types, seq_len)
    """
    divergences = np.zeros((pred.shape[0], pred.shape[1])) # output shape is (N, celltypes)
    for i in range(len(pred)): # for each ocr
        for j in range(pred.shape[1]): # for each ocr's celltype 
            divergences[i, j] = JSD_numpy(pred[i, j, :], obs[i, j, :])


    num_nan = np.sum(np.isnan(divergences))
    print("number nan JSD ", num_nan)
    return divergences

def plot_profile(obs_prof, pred_prof, title, out_path, obs_label='Observed', pred_label='Predicted', plot_bias=False, bias=None):
    """
    obs_prof and pred_prof are expected
    to be np arrays with a single dimension
    """
    # x = range(0, 250) # TODO just plot middle 250 bp
    plt.clf()
    # plt.bar(x, obs_prof[375: 625], color='b', label = 'observed')
    # plt.bar(x, pred_prof[375: 625], alpha=0.6, color='r', label = 'predicted')
    # plt.legend()
    # plt.title(title)
    # plt.savefig(out_path)
    # fig = plt.figure((0.05*np.shape(obs_prof)[-1],3.5), dpi = 200)

    fig = plt.figure(figsize = (max(0.05*np.shape(obs_prof)[-1],5.5),3.5), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x_axis = np.arange(np.shape(obs_prof)[-1])
    ax.bar(x_axis, obs_prof, label = obs_label, alpha = 0.9, color = 'blue')
    ax.bar(x_axis, pred_prof, label = pred_label, alpha = 0.9, color= 'red')
    combined_count = np.amin(np.array([obs_prof, pred_prof]),axis = 0) # max(0.05*np.shape(counts)[-1],5.5),3.5)
    ax.bar(x_axis, combined_count, label = 'Overlapping', alpha = 0.9, color= 'purple')
    if plot_bias:
        ax.bar(x_axis, bias, label = 'Bias', alpha = 0.5, color= 'dimgrey')
    ax.legend()
    ax.set_title(title)
    fig.savefig(out_path)
    plt.show()
    
def JSD_histogram(m_JSD:np.ndarray, model_title, pred_directory, filename='/JSD'):
    """
    expects a mean JSD for every OCR
    """
    plt.clf()
    plt.hist(m_JSD, bins=30)
    plt.ylabel("number of OCRs")
    plt.xlabel("JSD")
    plt.axvline(np.nanmean(m_JSD), color='r', linestyle='dashed', linewidth=2)
    plt.title("mean OCR JSD across cell types " + model_title)
    plt.savefig(pred_directory + filename + '.svg', format='svg')

def plot_loss_and_correlation(train_loss, val_loss, train_corr, val_corr, title, output_file_path):
    """This method takes in text file paths as input"""
    plt.clf()
    t_loss, v_loss = np.loadtxt(train_loss, delimiter=', '), np.loadtxt(val_loss, delimiter=', ')
    t_corr, v_corr = np.loadtxt(train_corr, delimiter=', '), np.loadtxt(val_corr, delimiter=', ')
    num_epochs = len(t_loss)

    # for i in range(data.shape[0] // num_epochs):
    #     plt.plot(np.arange(num_epochs + 1), data[i*num_epochs : i*num_epochs + num_epochs, 1])


    # plt.plot(t_loss[:, 0], t_loss[:, 1], 'b-', alpha=0.5, label='training loss')
    plt.plot(v_loss[:, 0], v_loss[:, 1], color='orange', label='validation loss')
    plt.axvline(x=np.argmin(v_loss[:,1])+1, color='orange', ls='--', label = 'min loss') 
    # plt.plot(t_corr[:, 0], t_corr[:, 1], 'b', alpha=0.5, label='training correlation')
    plt.plot(v_corr[:, 0], v_corr[:, 1], color='b', label='validation correlation')
    plt.axvline(x = np.argmax(v_corr[:,1]) + 1, color='b', ls='--', label = 'max correlation')
    plt.xlabel('Epoch', fontsize=9)
    # plt.xticks(np.arange(0, num_epochs + 1, 1))
    plt.ylabel('Loss / Correlation', fontsize=18)
    plt.title(title)
    plt.legend()
    plt.savefig((output_file_path+ "/" + title + ".png"))
    plt.close()

def plot_many_peaks_cells(obs_profile, pred_profile, cell_names, peak_names, model_title, out_dir):
    """
    pred_profile: numpy nd array of all the predicted profiles
    pred_profile: numpy nd array of all the observered profiles
    cell_names: numpy array of all cell names
    peak_names: numpy array of all peak names
    """
    cell_idx_list = [0, 1, 2, 4, 8, 10, 15, 20, 40, 60, 80, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     5, 5, 5, 5, 5, 5, 5, 5, 5]
    peak_idx_list = [16173, 9851, 1564, 7297, 4015, 10668, 7434, 11944, 3932, 1465, 1564, 
                     2000, 1000, 10000, 9000, 500, 10100, 11000, 3000, 15000,
                     100, 21000, 22000, 1, 18000, 15000, 1200, 1500, 1800]
    
    for i in range(len(cell_idx_list)):
        cell_idx = cell_idx_list[i]
        peak_idx = peak_idx_list[i]
        pred_prof = pred_profile[peak_idx, cell_idx, :]
        obs_prof = obs_profile[peak_idx, cell_idx, :]
        plot_profile(obs_prof[375:625], pred_prof[375:625], 
                     model_title + " " + cell_names[cell_idx] + " " + peak_names[peak_idx], # + " JSD=" + str(divergences[peak_idx, cell_idx]), 
                     out_dir + '/' + cell_names[cell_idx] + peak_names[peak_idx] + '_profile.png')
        
def plot_2aligned_profiles(obs_prof, pred_prof, title, outpath):
    """
    predicted profile and observed profile should ALREADY BE TRUCATED
    to the open chromatin region / desired region to plot
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(max(0.05*np.shape(obs_prof)[-1],5.5),7), dpi=200)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    x_axis = np.arange(np.shape(obs_prof)[-1])
    ax1.bar(x_axis, obs_prof, alpha = 0.9, color = 'c')
    ax1.set_title('True Profile')
    ax2.bar(x_axis, pred_prof, alpha = 0.9, color= 'orangered')
    ax2.set_title('Predicted Profile')
    # combined_count = np.amin(np.array([obs_prof, pred_prof]),axis = 0) # max(0.05*np.shape(counts)[-1],5.5),3.5)
    # ax.bar(x_axis, combined_count, label = 'Overlapping', alpha = 0.9, color= 'dimgrey')
    # ax.legend()
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(title)
    fig.savefig(outpath)
    plt.close()
    plt.clf()

def plot_3aligned_profiles(obs_prof, pred_prof, title, outpath):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(max(0.05*np.shape(obs_prof)[-1],5.5),10.5), dpi=200)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    x_axis = np.arange(np.shape(obs_prof)[-1])
    ax1.bar(x_axis, obs_prof, alpha = 0.9, color = 'c')
    ax1.set_title('True Profile')
    ax2.bar(x_axis, pred_prof, alpha = 0.9, color= 'orangered')
    ax2.set_title('Predicted Profile')

    ax3.bar(x_axis, obs_prof, label = 'Observed', alpha = 0.9, color = 'c')
    ax3.bar(x_axis, pred_prof, label = 'Predicted', alpha = 0.9, color= 'orangered')
    combined_count = np.amin(np.array([obs_prof, pred_prof]),axis = 0) # max(0.05*np.shape(counts)[-1],5.5),3.5)
    ax3.bar(x_axis, combined_count, label = 'Overlapping', alpha = 0.9, color= 'black')
    ax3.legend()
    ax3.set_title('Overlap')

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(title)
    fig.savefig(outpath + '.svg', format='svg')
    plt.clf()

def scatter_heat(x, y, x_label, y_label, outpath, title=''):
    print(outpath)
    ab = np.vstack([x,y])
    colors = gaussian_kde(ab)(ab)
    cmap = 'viridis'
    plt.clf()
    plt.scatter(x ,y, c = colors, cmap=cmap, s=50,alpha=.7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(outpath + '.svg', format='svg')
    plt.close()

def scatter_with_diagonal(x, y, x_label, y_label, outpath):
    print(outpath)
    vals = np.vstack([x, y])
    colors = gaussian_kde(vals)(vals)
    cmap = 'viridis'
    
    # Set figure size to make it square
    plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.gca()  # Get the current axis
    
    ax.scatter(x, y, c=colors, cmap=cmap, s=170, alpha=0.7)
    
    lim = [np.amax(np.amin(vals, axis=1)), np.amin(np.amax(vals, axis=1))]
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color='maroon')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make axes have the same length
    xlim = [np.amin(vals), np.amax(vals) + 0.01] 
    ylim = [np.amin(vals), np.amax(vals) + 0.01] 
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    pearson_corr = round(pearsonr(vals[0], vals[1])[0], 2)

    # Create an outlined text box for the Pearson correlation coefficient
    text_box = FancyBboxPatch((0.7, 0.02), 0.2, 0.1, boxstyle="round, pad=0.05", edgecolor="black", facecolor="white")
    ax.add_patch(text_box)
    
    # Add Pearson correlation coefficient as a label inside the text box
    ax.text(0.8, 0.07, 'R: ' + str(pearson_corr), transform=ax.transAxes, fontsize=12, horizontalalignment='right')
    
    # Adjust font size for axis labels and ticks
    plt.rc('font', size=14)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    plt.savefig(outpath + '.svg', format='svg')
    plt.close()





def sara_graphs(obs_profile, pred_profile, cell_names, peak_names, out_dir, model_name):
    eps = 1e-8
    # FOR SAMPLE:
    # cell_idx_list = [0, 1]
    # peak_idx_list = [0, 100]
    print("outdir", out_dir)

    cell_idx_list = [0, 1, 2, 4, 8, 10, 15]
    peak_idx_list = [16173, 9851, 1564, 7297, 4015, 10668, 7434]

    pred_profile = (pred_profile[:, :, 375:625] + eps) / np.sum(pred_profile[:, :, 375:625] + eps, axis=-1, keepdims=True)
    obs_profile = (obs_profile[:, :, 375:625] + eps) / np.sum(obs_profile[:, :, 375:625] + eps, axis=-1, keepdims=True)
    print("pred profile shape HERE", pred_profile.shape)
    print("obs profile shape",  obs_profile.shape)
    
    for i in range(len(cell_idx_list)):
        cell_idx = cell_idx_list[i]
        peak_idx = peak_idx_list[i]
        pred_prof = (pred_profile[peak_idx, cell_idx, :]) 
        obs_prof = (obs_profile[peak_idx, cell_idx, :])
    
        # we are going to normalize these both? is that necessary. Should not be for observed because 
        # that is already normalized to the center 250 bp
        # however, the predicted one was softmaxed over the entire 1000 bp, so it needs to be normalized 
        print(obs_prof.shape, print(pred_prof.shape))
        jsd = JSD_numpy(pred_prof, obs_prof)
        cell = cell_names[cell_idx]
        peak = peak_names[peak_idx]
        title =  cell + " " + peak
        print(cell, peak, "JSD:", jsd)
        plot_2aligned_profiles(obs_prof, pred_prof, 
                     title, out_dir + '/' + cell + peak + '_pred_obs_prof.svg')
        
        # plot_3aligned_profiles(obs_prof, pred_prof, 
        #         title, out_dir + '/' + model_name + cell + peak + '_3prof.pdf')
        scatter_with_diagonal(obs_prof, pred_prof, "True Profile", "Predicted Profile", os.path.join(out_dir, cell + peak + 'scatter.svg'))

def bp_corr_histogram(bp_corr, cell_names, cell, model_title, outpath):
    """
    This method plots a histogram of base-pair correlations    
    for either a single cell type or all cell types

    bp_corr: numpy array of correlations for each OCR and cell type
        expected shape: (#OCRs, #cell_types)
    cell_names: numpy array of cell names - assumes same order as in the bp_corr
    cell: the name of a specific cell that should be plotted. 
    outpath should contain the full path of the saved graph except for the file type indicator
    """

    cell_idx = np.where(cell_names == cell)
    correlations = bp_corr[:, cell_idx].flatten()
    plt.clf()
    plt.hist(correlations, bins=30)
    plt.ylabel("Number of OCRs")
    plt.xlabel("Base-Pair Pearson Correlation")
    plt.axvline(np.nanmean(correlations), color='r', linestyle='dashed', linewidth=2)
    plt.title(model_title +  " " + cell + " Base Pair Correlations")
    plt.savefig(outpath + '.svg', format='svg')
    plt.close()

def corr_histogram(correlations, model_title, outpath, avg_celltypes=False):
    """
    plot histogram of correlation 
    if avg_celltypes = true, then it plots the avg correlation accross celltypes for each ocr
    otherwise plots for all OCR and celltypes
    """
    plt.clf()
    if avg_celltypes:
        correlations = np.mean(correlations, axis=-1)
        plt.ylabel("# OCRs")
        plt.title(model_title +  "Avg Total Count Pearson Correlations Across Celltypes")
    else:
        correlations = correlations.flatten()
        plt.ylabel("Frequency")
        plt.title(model_title +  "Total Count Pearson Correlations")
    plt.hist(correlations, bins=30)
    plt.xlabel("OCR Total Count Prediction Pearson Correlation")
    plt.axvline(np.nanmean(correlations), color='r', linestyle='dashed', linewidth=2)
    plt.savefig(outpath + '.svg', format='svg')
    plt.close()
    

def get_mse(scalar_obs, scalar_pred):
    """
    returns the mse for each OCR and celltype
    """
    squared_diff = np.square(scalar_obs - scalar_pred)
    mse = np.mean(squared_diff, axis=-1) # takes the mean ALONG THE SEQUENCE
    return mse

def plot_squared_error_vs_bp_corr(scalar_obs, scalar_pred, bp_corr, cell_names, cell, model_title, outpath):
    """
    outpath should contain the full path of the saved graph except for the file type indicator
    """
    squared_diff = np.square(scalar_obs - scalar_pred)
    cell_idx = np.where(cell_names == cell)
    scatter_heat(bp_corr[:, cell_idx].flatten(), squared_diff[:, cell_idx].flatten(), 
                 'Base-Pair Pearson Correlation', 'Squared Error',
                 outpath)
    scatter_heat(bp_corr[:, cell_idx].flatten(), np.log(squared_diff[:, cell_idx].flatten()), 
                 'Base-Pair Pearson Correlation', 'Log Squared Error',
                 outpath + "_log")

def squared_error_histogram(scalar_obs, scalar_pred, cell_names, cell, model_title, outpath):
    squared_diff = np.square(scalar_obs - scalar_pred)
    cell_idx = np.where(cell_names == cell)
    plt.clf()
    plt.hist(squared_diff[:, cell_idx].flatten(), bins=30)
    plt.xlabel("Squared error of scalar prediction")
    plt.axvline(np.nanmean(squared_diff[:, cell_idx].flatten()), color='r', linestyle='dashed', linewidth=2)
    plt.title(model_title +  " " + cell + " Squared Error")
    plt.savefig(outpath + '.svg', format='svg')
    plt.close()

    # plot log of errors
    plt.clf()
    plt.hist(np.log(squared_diff[:, cell_idx].flatten()), bins=30)
    plt.xlabel("Log of squared error of scalar prediction")
    plt.axvline(np.nanmean(np.log(squared_diff[:, cell_idx].flatten())), color='r', linestyle='dashed', linewidth=2)
    plt.title(model_title +  " " + cell + " Log Squared Error")
    plt.savefig(outpath + '_log.svg', format='svg')
    plt.close()

def plot_scalar_obs_v_pred(scalar_obs, scalar_pred, cell_names, cell, model_title, outpath):
    """
    if cell is 'none', will plot all cells
    otherwise will plot specific cell
    """
    if cell != 'none':
        cell_idx = np.where(cell_names == cell)
        scalar_obs = scalar_obs[:, cell_idx].flatten()
        scalar_pred = scalar_pred[:, cell_idx].flatten()
        title = model_title + " " + cell + " Total Count Prediction vs Observed"
    else:
        scalar_obs = scalar_obs.flatten()
        scalar_pred = scalar_pred.flatten()
        title = model_title + " Total Count Prediction vs Observed"
    plt.clf()
    scatter_heat(scalar_obs, scalar_pred, 'Total OCR Counts', 'Predicted Total OCR Counts', outpath, title=title)

def plot_profile_on_an_axis(ax, obs_profile, pred_profile, cell_name, plot_bias=False, bias=None):
    x_axis = np.arange(len(obs_profile))
    ax.bar(x_axis, pred_profile, label=f"Predicted", alpha=0.9, color='red', width=1)
    ax.bar(x_axis, obs_profile, label=f"Observed", alpha=0.9, color='blue', width=1)
    combined_count = np.amin(np.array([obs_profile, pred_profile]), axis=0)
    ax.bar(x_axis, combined_count, label='Overlapping', alpha=0.9, color='fuchsia', width=1)
    if plot_bias:
        ax.bar(x_axis, bias, label='Bias', alpha=0.5, color='dimgrey', width=1)
    ax.set_title(f"OCR Profile: {cell_name}")
    ax.legend()

# Note: this method does not currently actually support bias- because it is plotting the probability distributions
# If modified to plot the bp counts, it could plot bias
def plot_top_ocr_profiles(model, dataloader, ocr_celltypes, compare_celltypes, cell_names, peak_names, outdir, ocr_start=375, ocr_end=625, plot_bias=True):
    DEVICE = get_least_utilized_gpu()
    print(DEVICE)
    model.to(DEVICE)
    
    # convert ocr_celltypes into a set
    ocr_celltypes = set(ocr_celltypes) 
    batch_size = dataloader.batch_size
    cumulative_index = 0

    for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(dataloader):
        print("batch num", i)
        seqs, bias, total_counts, bp_counts = seqs.to(DEVICE), seq_bias.to(DEVICE), total_counts.to(DEVICE), bp_counts.to(DEVICE)
        bias = torch.squeeze(seq_bias).to(DEVICE) # the squeezing to work with current head setup
        pred_profile, pred_scalar = model(seqs, bias) # we need to make predictions on only the seqs with the relevant ocrs, but that means we have to be careful about indexing, so we are just going to predict on all

        # Convert the tensors to NumPy arrays
        total_counts_np = total_counts.cpu().numpy()
        bp_counts_np = bp_counts.cpu().numpy()
        pred_profile_np = pred_profile.cpu().detach().numpy()
        bias_np = bias.cpu().detach().numpy()

        if (np.array_equal(np.sum(bp_counts_np, axis=-1),total_counts_np)):
            print("Total counts are the sum of base pair counts")
        
        # Find the OCR celltype for each peak 
        peak_ocr_celltype_indices = np.argmax(total_counts_np, axis=1)
        current_ocr_cells_for_each_peak = cell_names[peak_ocr_celltype_indices]

        # Find all the ocr_celltypes in current_ocr_cells and remove them from ocr_celltypes set
        found_ocr_celltypes = set(current_ocr_cells_for_each_peak) & ocr_celltypes
        ocr_celltypes -= found_ocr_celltypes

        # plot each of the found_ocr_celltypes 
        for cell in found_ocr_celltypes:
            # Find the first peak that the cell is an ocr of:
            peak_idx = np.where(current_ocr_cells_for_each_peak == cell)[0][0]
            peak_name = peak_names[cumulative_index + peak_idx]
            current_bias = bias_np[peak_idx]
            current_bias = softmax_numpy(current_bias) # we are doing things in terms of distributions

            # Find the index of the current cell
            curr_cell_index = np.where(cell_names == cell)[0]


            # Get the predicted and observed profiles for the current cell
            curr_cell_pred_profile = np.squeeze(pred_profile_np[peak_idx, curr_cell_index])
            # The observed profile has to be converted to a probability
            curr_cell_obs_bp_counts = np.squeeze(bp_counts_np[peak_idx, curr_cell_index])
            curr_cell_obs_profile = curr_cell_obs_bp_counts / np.sum(curr_cell_obs_bp_counts)
            
            # get the center ocr
            curr_cell_pred_profile, curr_cell_obs_profile = curr_cell_pred_profile[ocr_start: ocr_end], curr_cell_obs_profile[ocr_start:ocr_end]

            # Create a list of compare celltypes, excluding the current cell
            filtered_compare_celltypes = [c for c in compare_celltypes if c != cell]

            # Create a figure with subplots for the current OCR celltype and compare celltypes
            num_cols = len(filtered_compare_celltypes) + 1
            fig, axs = plt.subplots(1, num_cols, figsize=(num_cols*5, 3), dpi=200)
            max_y = max(np.max(curr_cell_obs_profile), np.max(curr_cell_pred_profile))

            # Plot the current cell in the first subplot
            plot_profile_on_an_axis(axs[0], curr_cell_obs_profile, curr_cell_pred_profile, cell, plot_bias=plot_bias, bias=current_bias[ocr_start:ocr_end])


            # Plot the compare celltypes in the remaining subplots
            for col, compare_cell in enumerate(filtered_compare_celltypes, start=1):
                compare_cell_index = np.where(cell_names == compare_cell)[0][0]
                compare_cell_pred_profile = np.squeeze(pred_profile_np[peak_idx, compare_cell_index])
                compare_cell_obs_bp_counts= np.squeeze(bp_counts_np[peak_idx, compare_cell_index])
                compare_cell_obs_profile = compare_cell_obs_bp_counts / np.sum(compare_cell_obs_bp_counts)
                compare_cell_pred_profile, compare_cell_obs_profile = compare_cell_pred_profile[ocr_start: ocr_end], compare_cell_obs_profile[ocr_start:ocr_end]

                plot_profile_on_an_axis(axs[col], compare_cell_obs_profile, compare_cell_pred_profile, compare_cell, plot_bias=plot_bias, bias=current_bias[ocr_start:ocr_end])
                max_y = max(max_y, np.max(compare_cell_obs_profile), np.max(compare_cell_pred_profile))
            
            for i in range(num_cols):
                axs[i].set_ylim([0, max_y])

            # Adjust the spacing between subplots
            plt.tight_layout()
            
            # Save the figure
            out_path = os.path.join(outdir, f"ocr_profile_{cell}_peak_{peak_name}.png")
            print(out_path)
            fig.savefig(out_path)
            plt.close(fig)
        
        cumulative_index += batch_size

        # Break the loop if all OCR celltypes have been found
        if not ocr_celltypes:
            break



def main():
    """
    This script plots results for a single model at a time
    """
    parser = argparse.ArgumentParser(description="Analyze and plot model predictions.")
    parser.add_argument("--model_directory", type=str, required=True, help="Path to the directory containing predictions.")
    parser.add_argument("--data_info_file", type=str, required=True, help="Path to the data info file.")
    parser.add_argument("--model_title", type=str, required=True, help="Model name and timestamp.")
    parser.add_argument("--n_filters", type=int, required=True, help="Number of filters.")
    parser.add_argument("--cell_names", type=str, required=True, help="Path to the cell names file.")
    parser.add_argument("--peak_names", type=str, required=True, help="Path to the peak names file. This must MATCH the eval set")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--save_predictions", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Whether to save all predictions. If not provided defaults to false")
    parser.add_argument("--run_analysis", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Whether to analyze the models predictions with scalar correlation, bp correlation etc.")
    parser.add_argument("--analyzed_data_file", type=str, required=False, help="If given, will not compute new analysis of the model. Path to a npz file conting already computed analysis results for the model. Must contain files corresponding to the names: ")
    parser.add_argument("--example_profiles", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Whether to make example profile plots. Defaults to false if argument not provided.")
    parser.add_argument("--example_profiles_eval_set", type=str, required=False, default="Validation", help="Must be either Train or Validataion. Test not currently supported")
    parser.add_argument("--example_profiles_ocr_celltypes", nargs='+', required=True, help="space separated list of OCR celltypes to plot.")
    parser.add_argument("--example_profiles_compare_celltypes", nargs='+', required=True, help="space separated list of celltypes to compare against OCR celltypes.")
    # add an argument for the celltypes 

    args = parser.parse_args()

    model_path = os.path.join(args.model_directory, 'best_model')
    cell_names = np.load(args.cell_names)
    n_celltypes = len(cell_names)

    # Load model and make predictions on validation set
    if args.save_predictions: 
        predictions_filepath = os.path.join(args.model_directory, 'predictions.npz')
        if os.path.isfile(predictions_filepath):
            print("predictions.npz already exists")
            data = np.load(predictions_filepath)
            print("loaded npz")
            print(data.files)
            scalar_obs, scalar_preds, profile_obs, profile_pred = data['scalar_obs'], data['scalar_preds'], data['profile_obs'], data['profile_pred']
        else:
            scalar_obs, scalar_preds, profile_obs, profile_pred = get_predictions(model_path,
                                                                                args.data_info_file,
                                                                                n_celltypes,
                                                                                args.n_filters,
                                                                                args.batch_size,
                                                                                save_pred=True)
            np.savez(os.path.join(predictions_filepath), scalar_obs=scalar_obs, scalar_preds=scalar_preds, profile_obs=profile_obs, profile_pred=profile_pred)

        # TODO reorganize where this call goes
        sara_graphs(profile_obs, profile_pred, cell_names, args.peak_names, args.pred_directory, args.model_name)
        print("sara graphs done")

    
    # if args.run_analysis:
    # Get the pre-calculated metrics:
    if args.analyzed_data_file:
        print("using existing", args.analyzed_data_file)
        analysis = np.load(args.analyzed_data_file)
    else:
        # Call eval_model.py to create analysis.npz
        eval_model(
            saved_model_path=model_path,
            n_celltypes=n_celltypes,
            n_filters=args.n_filters,
            infofile_path=args.data_info_file,
            get_scalar_corr=True,
            get_profile_corr=True,
            get_jsd=True,
            eval_set='validation',
            output_dir=args.model_directory
        )
        # Load the generated analysis.npz file
        analysis = np.load(os.path.join(args.model_directory, 'validation_analysis.npz'))

    scalar_corrs, bp_corrs, jsd = analysis['scalar_corr'], analysis['profile_corr'], analysis['jsd']
    ocr_bp_corrs, ocr_jsd = analysis['ocr_profile_corr'], analysis['ocr_jsd']
    top_ocr_bp_corrs, top_ocr_jsd = analysis['top_ocr_profile_corr'], analysis['top_ocr_jsd']

    peak_names = np.memmap(args.peak_names, dtype='<U26', shape=((scalar_corrs.shape[0])))
    if args.example_profiles:
        # This implementation currentlyl does not support binning
        model = load_model( 
            saved_model_path=model_path,
            n_celltypes=n_celltypes,
            n_filters=args.n_filters, 
            bin_size=1)
        train_loader, val_loader, test_loader = load_data(info_file=args.data_info_file, batch_size=100)
        if args.example_profiles_eval_set == "Train":
            dataloader = train_loader
        elif args.example_profiles_eval_set == "Validation":
            dataloader = val_loader
        else:
            print("INVALID plot_example_profiles_eval_set argument")
            exit(1)
        # peak_names = np.load(args.peak_names, allow_pickle=True)
        print("peak_names check", peak_names[:5])
        plot_top_ocr_profiles(model, dataloader, args.example_profiles_ocr_celltypes, args.example_profiles_compare_celltypes, cell_names, peak_names=peak_names, outdir=args.model_directory)
        
        
        

    # load the memmap of saved validation data 
    # scalar_obs = np.memmap(obs_tcounts_path, dtype='float32', shape=(obs_n_peaks, len(cell_names)))

    # plot observed vs predicted
    # scatter_plot(scalar_obs, scalar_preds, 'Observed OCR Total Counts', 'Predicted OCR Total Counts',
    #                model_title + " OCR Total Count", pred_directory + '/val_scatter.svg')

    # correlations = plot_cors(scalar_obs, scalar_preds, pred_directory + '/')
    # average correlation for 90 cell types
    # c_m = np.mean(correlations, axis=0)

    # load profile predictions
    # eps = 1e-8
    # bp_obs =  np.memmap(obs_profile_path, dtype='float32', shape=(obs_n_peaks, len(cell_names), seq_len))
    # profile_obs = (profile_obs + eps) / np.sum(profile_obs + eps, axis=-1, keepdims=True) # get the distributions NOTE this will need to be added back in in some form, if you want to run things below. saras graphs is now doing it internally
    # profile_pred = np.load(pred_directory + '/predictions.npz', allow_pickle=True)['profile']
    # divergences = get_JSD(profile_pred, profile_obs)
    # np.save(pred_directory + '/JSD', divergences) 

    # cell_type = 'MF.PC' # FOR TESTING
    # bp_corr_histogram(bp_corrs, cell_names, cell_type, model_name, 'bp_corr_histogram')
    # plot_squared_error_vs_bp_corr(scalar_obs, scalar_preds, bp_corrs, cell_names, cell_type, model_name, 'squared_error_scatter')
    # squared_error_histogram(scalar_obs, scalar_preds, cell_names, cell_type, model_name, 'squared_error_histogram')
    

    # cell_type = 'B.FrE.BM'
    # # for testing use below
    # # cell_type = 'MF.PC'
    # # for a subset of cells and OCRs plots the profile predictions and a scatter plot of profile predictions

    # # plot total counts 
    # # plot_scalar_obs_v_pred(scalar_obs=scalar_obs, scalar_pred=scalar_preds, cell_names=cell_names,  # THIS PLOT TAKES SO LONG TO RUN - haven't been able to make it work yet
    # #                        cell='none', model_title=model_name, outpath=os.path.join(pred_directory, 'scalar_scatter'))
    # plot_scalar_obs_v_pred(scalar_obs=scalar_obs, scalar_pred=scalar_preds, cell_names=cell_names, 
    #                        cell=cell_type, model_title=model_name, outpath=os.path.join(pred_directory, cell_type + 'scalar_scatter'))
    # print("scalar obs v pred done")
    # bp_corr_histogram(bp_corrs, cell_names, cell_type, model_name, os.path.join(pred_directory, 'bp_corr_histogram'))
    # bp_corr_histogram(ocr_bp_corrs, cell_names, cell_type, model_name + " OCR ", os.path.join(pred_directory, 'ocr_bp_corr_histogram'))
    # print("bp corr histograms done")

    # plot_squared_error_vs_bp_corr(scalar_obs, scalar_preds, bp_corrs, cell_names, cell_type, model_name, os.path.join(pred_directory, 'squared_error_scatter'))
    # plot_squared_error_vs_bp_corr(scalar_obs, scalar_preds, ocr_bp_corrs, cell_names, cell_type, model_name + ' OCR', os.path.join(pred_directory, 'squared_error_ocr_scatter'))

    # squared_error_histogram(scalar_obs, scalar_preds, cell_names, cell_type, model_name, os.path.join(pred_directory, 'squared_error_histogram'))


    # # plot mean JSD for an OCR across cell types 
    # # get the mean JSD across the cell types
    # m_JSD = np.mean(ocr_jsd,axis=1)
    # JSD_histogram(m_JSD, model_title, pred_directory, filename='/ocr_jsd_histogram') # expects OCR JSD
    # print("jsd histogram done")

    # # plot scalar correlation correlation
    # corr_histogram(scalar_corrs, model_title, os.path.join(pred_directory, 'scalar_corr_histogram'), avg_celltypes=False)
    # corr_histogram(scalar_corrs, model_title, os.path.join(pred_directory, 'scalar_corr_histogram_avg'), avg_celltypes=True)

    # Plot the B-profile (a profile with good bias that we use to compare across models)
    # currently using celltype from MF.PC
    # b_peak = 'ImmGenATAC1219.peak_348578'
    # b_cell = 'MF.PC'

    # print("where the peak_names thing is equal", np.argwhere(peak_names == b_peak))
    # b_ocr_idx = np.argwhere(peak_names == b_peak).item() # TODO THIS FAILS BECAUSE PEAK NAME IS NOT IN VAL SET
    # b_cell_idx = np.argwhere(cell_names == b_cell).item()
    # print("b_profile peak name", (peak_names[b_ocr_idx]))
    # print(cell_names[b_cell_idx], b_cell_idx)
    # b_pred_prof = profile_pred[b_ocr_idx, b_cell_idx, :]
    # b_obs_prof = profile_obs[b_ocr_idx, b_cell_idx, :]
    # title = b_cell + b_peak 
    # b_JSD = divergences[b_ocr_idx, b_cell_idx].item()
    # print(b_JSD)
    # # plot the middle 250 bp (assuming ~1000 bp total sequence)
    # plot_profile(b_obs_prof[375:625], b_pred_prof[375:625], title + " JSD =" + str(round(b_JSD, 3)), 
    #              pred_directory + '/' + b_cell + b_peak + '_profile.png')


    # plot one of the the best profiles with a barplot
    # find the ocr for each cell type with the lowest divergence 
    # ocr_indices  = np.argmin(divergences, axis=0)
    # print("ocr indices shape", ocr_indices.shape)
    # # choose one of the cell types randomly 
    # cell_idx = np.random.randint(0, len(cell_names))
    # ocr_idx = ocr_indices[cell_idx]
    # print("ocr idx", ocr_idx)


    # chosen_jsd = divergences[ocr_idx, cell_idx]
    # print("chosen jsd is ", chosen_jsd)

    # cell_type = cell_names[cell_idx] # the 0,1 entry of the argwhere is celltype idx
    # print(cell_type)
    # # get the profile prediction and best profile observation
    # pred_prof = profile_pred[ocr_idx, cell_idx, :]
    # obs_prof = profile_obs[ocr_idx, cell_idx, :]
    # plt.clf()
    # # plot the entire sequence
    # plot_profile(obs_prof, pred_prof, model_title + " " + cell_type + " JSD =" + str(round(chosen_jsd, 3)), pred_directory + '/good_profile_long')
    # # plot the middle 250 bp (assuming ~1000 bp total sequence)
    # plot_profile(obs_prof[375:625], pred_prof[375:625], model_title + " " + cell_type + " JSD =" + str(round(chosen_jsd, 3)), pred_directory + '/good_profile_middle')

    # # plot a profile section where there are bias counts but no observed counts
    # cell = "MF.RP.Sp"
    # peak = "ImmGenATAC1219.peak_312006"
    # x_cell_idx = np.argwhere(cell_names == cell).item() NOTE: THIS FAILS BECAUSE IT is not in val set -- I think
    # x_peak_idx = np.argwhere(peak_names == peak).item()
    # print("peak idx found", x_peak_idx)
    # x_pred_prof = profile_pred[x_peak_idx, x_cell_idx, :]

    # plot_profile(x_obs_prof[0:625], x_pred_prof[0:625], model_title + " " + cell + " " + peak, 
    #              pred_directory + '/' + cell + peak + '_profile.png')
    

    # # # plot the middle 250 bp (assuming ~1000 bp total sequence)


    # # plot several peaks with a high number of total counts in the validation set


    # # plot observed vs predicted scalar counts for GN.Thio.PC
    # g_idx = 5
    # scatter_plot(scalar_obs[:, 5], scalar_preds[:, 5], 
    #              'Observed OCR Total Counts', 'Predicted OCR Total Counts', 
    #              model_title + " GN.Thio.PC Total Count", pred_directory + '/val_GN.Thio.PC.scatter.png')
    

if __name__ == "__main__":
    main()

