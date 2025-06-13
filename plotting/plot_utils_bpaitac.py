import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_observed
from functions import coefficient_of_variation

# This method returns a mask over celltypes: returns indices
#  for the one with the largest value for each peak
def ocr_mask(total_counts: torch.Tensor):
    max_celltype_indices = torch.argmax(total_counts, dim=1)
    return max_celltype_indices

# expects data to be in shape #peaks, seq_len, celltypes
def get_only_ocrs(data:torch.Tensor, total_counts:torch.Tensor):
    mask = ocr_mask(total_counts=total_counts)
    out = data[torch.arange(data.size(0)), mask]
    return out


def get_cell_index(celltypes:np.ndarray, target_celltype: str):
    index = np.where(celltypes == target_celltype)[0]
    if len(index) > 0:
        print(f"The first index of '{target_celltype}' is: {index[0]}")
    else:
        print(f"'{target_celltype}' not found in the array.")
    return index[0]

def histogram(data:np.ndarray, x_label:str, output_path:str, title:str=None, format:str='svg', save_fig:bool=True):
    plt.clf()
    plt.hist(data, bins=30)
    plt.xlabel(x_label)
    plt.axvline(np.nanmean(data), color='r', linestyle='dashed', linewidth=2)
    plt.title(title)
    if save_fig:
        plt.savefig('{}.{}'.format(output_path, format), format=format)
    plt.show()
    plt.close()




# plots the "true ocrs" as defined by the ocr threshold
# can specify a specific celltype too
def plot_metric_for_celltype_ocrs(
    pred, # expected shape: (#peaks, sequence_len, #celltypes)
    obs, 
    total_counts, 
    n_examples: int, 
    ocr_threshold:int, 
    out_dir: str,
    celltypes:np.ndarray=None, # not tested yet
    celltype_to_plot:str=None,
):
    if celltype_to_plot is not None:
        cell_index = get_cell_index(celltypes, celltype_to_plot)
        # get the data for only the relevant celltype
        pred = pred[:, :, cell_index]
        obs = obs[:, :, cell_index]
        total_counts = total_counts[:, cell_index]
    
    # find the ocrs for that celltype
    ocr_mask = ocr_mask(total_counts, ocr_threshold)
    print("mask shape", ocr_mask.shape)
    pred_ocr = pred[ocr_mask]
    obs_ocr = obs[ocr_mask]
    print("obs_ocr shape", obs_ocr.shape, "pred_ocr shape", pred_ocr.shape)


def get_coefficient_of_variance_quartile_array(info_file):
    total_counts = load_observed(info_file, dataset_type='test', data_name='total_counts')
    cv = coefficient_of_variation(total_counts)

    q1, q2, q3 = np.percentile(cv, [25, 50, 75])
    print(q1, q2, q3)

    # Create an array of the same length as cv, initialized with zeros
    quartile_array = np.zeros_like(cv, dtype=int)

    # Assign quartile numbers
    quartile_array[cv < q1] = 1
    quartile_array[(cv >= q1) & (cv < q2)] = 2
    quartile_array[(cv >= q2) & (cv < q3)] = 3
    quartile_array[cv >= q3] = 4
    return quartile_array


