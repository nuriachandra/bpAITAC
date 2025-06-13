from plotting.combine_results import combine_trial_metrics, find_files
import os
from eval_model import get_model_structure
import numpy as np
from utils.inference_utils import load_names
from plotting.plot_utils_bpaitac import histogram
from collections import defaultdict
import pandas as pd


"""
Data Structure:
- For each region in the test data we need to be able to look at the individual pearson correlation
    from each trial
- Hmmm Do I already have that somewhere? or maybe it's best to just run it through the model one region at a
    time and compute it again. 
"""
def get_trial_metrics(model_dir, info_path, n_celltypes, n_filters, cur_bin_size, bin_pooling_type,
                          scalar_head_fc_layers, 
                          eval_set,
                          new_analysis:bool,
                          analysis_file_name,
                          model_structure,
                          ):
    """
    returns a npz file with all the metrics in it- separated in the last dim by trial
        use .files to see all the file names
    model_dir: the directory (usually for a specific lambda value). Will calculate
        metrics for all best_model within the sub-directories of this directory
        This is also where the metrics will be saved
    new_analysis: boolean indicating if the combined results should be saved
        Note, this will NOT run new_analysis on trials where analysis has already been saved
        This is good to do if you have new trials that you want to include,
        or you are using a new eval_set that has not been used yet
    analysis_file_name: this is the file name that will be used for saving individual trial's 
        best model metrics. This name will also be used to search for existing metic files.
    save_combined: boolean indicating whether the combined metrics (not separated by trials) 
        should be saved. This is a good idea if this is the first time you are getting
        metrics for the saved models within model_dir
    model_type: either 'BPnetRep' or 'BPcm'
    """
    paths = find_files(model_dir, 'best_model')
    print(paths)

    # Get the metrics for test data 
    # We want to start out by trying with a sample dataset 
    combine_trial_metrics(paths, info_path, n_celltypes, n_filters, bin_size=cur_bin_size, 
                          bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers, 
                          save_combined=new_analysis, 
                          save_trials=True, 
                          save_combined_dir=model_dir, 
                          eval_set=eval_set, 
                          analysis_file_name=analysis_file_name,
                          only_one_trial=False,
                          model_structure=model_structure
                        )
    return np.load(os.path.join(model_dir, 'analysis_' + eval_set + '_trials.npz'))

def identify_regions(arr1, arr2):
    """
    this method has to take in two numpy arrays of shape (n_regions, n_trials) and 
    returns the indices the regions which are higher in all of the trials of the
    second array than the max of the first array over all trials
    max_trials(arr1) < min_trials(arr2)
    """
        # Check if the input arrays have the correct shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Input arrays must have the same shape. array1 has shape {arr1.shape}, array2 has shape {arr2.shape}. There are different numbers of trials here.")
    
    # Find the maximum value across all trials for each region in arr1
    max_arr1 = np.max(arr1, axis=1)
    
    # Find the minimum value across all trials for each region in arr2
    min_arr2 = np.min(arr2, axis=1)
    
    # Compare the maximum of arr1 with the minimum of arr2 for each region
    higher_regions = min_arr2 > max_arr1
    
    # Get the indices of the regions that satisfy the condition
    indices = np.where(higher_regions)[0]
    return indices
     
def summary_stats(arr1, arr2, indices, outdir, metric_name, label1, label2):
    """
    Where indices are the indices of arr1 and arr2 (in dim=0) where
    arr1 < arr2 for all trials

    computes the average difference between predicted and actual arrays 
    
    """
    if arr1.shape != arr2.shape:
        raise ValueError(f"Input arrays must have the same shape. IF they don't have the same number of trials the averaging won't work array1 has shape {arr1.shape}, array2 has shape {arr2.shape}")
    
    print("Number of regions:", len(indices))
    print("Fraction of regions with increase", len(indices)/len(arr1))

    # metrics around the differences
    diff = arr2 - arr1
    avg_diff = np.mean(diff, axis=1)
    print(avg_diff.shape)
    print("Mean difference", np.mean(diff))
    print("Mean difference in improved regions", np.mean(diff[indices]))

    x_axis_label = " ".join(['difference in', metric_name, label1, label2])
    # difference histogram
    histogram(diff.flatten(), x_axis_label, os.path.join(outdir, 'difference_histogram'+ '_' + label1+'_' + label2), format='png')
    # difference averaged over trials
    histogram(avg_diff, x_axis_label, os.path.join(outdir, 'difference_histogram_avg'+ '_' + label1+'_' + label2), format='png')


    # improved difference
    histogram(diff[indices].flatten(), x_axis_label, os.path.join(outdir, 'improved_region_difference_histogram'+ '_' + label1+'_' + label2), format='png')
    # difference averaged over trials
    histogram(avg_diff[indices], x_axis_label, os.path.join(outdir, 'improved_region_difference_histogram_avg'+ '_' + label1+'_' + label2), format='png')


def main():
    n_celltypes = 90
    n_filters = 300
    n_trials_to_use = None

    eval_set = 'validation'
    info_path = '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt'
    names = load_names('/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt', eval_set)
    
    # For BP71
    model1_dir = '/data/nchand/analysis/BPnetRep/BP71_L0_0/'
    model2_dir = '/data/nchand/analysis/BPnetRep/BP71_L-1_1/'
    output_dir = '/data/nchand/analysis/BPnetRep/BP71_analysis'
    label1='l=0.0'
    label2='l=0.1'
    model_type = 'BPnetRep'
    n_trials_to_use = 5 # TODO update when I have more

    # For BP68
    # model1_dir = '/data/nchand/analysis/BPcm/BP68_L0_0/'
    # model2_dir = '/data/nchand/analysis/BPcm/BP68_L-1_5/'
    # output_dir = '/data/nchand/analysis/BPcm/BP68_analysis'
    # difference_label ='Difference in Pearson correlation at lambda=0.5 and lambda=0.0'
    # n_trials_to_use = 5
    # label1='l=0.0'
    # label2='l=0.5'
    # model_type = 'BPcm'


    os.makedirs(output_dir, exist_ok=True)


    analysis_file_name = 'validation_analysis.npz'
    model_structure = get_model_structure(model_type, n_filters, n_celltypes)
    new_analysis = True

    bin_size= 1
    bin_pooling_type = None
    scalar_head_fc_layers = 1
    

    """
    Find the regions which are better predicted in model 2 than model 1

    model1_dir the parent directory of all the trials of model 1
        this dir will contain an 'analysis_' + eval_set + '_trials.npz' file
        If it does not already contain one, one will be created
    model2_dir the parent directory of all the trials of model 2
        this dir will contain an 'analysis_' + eval_set + '_trials.npz' file 
    """
    model1_metrics = get_trial_metrics(model1_dir, info_path, n_celltypes, n_filters, bin_size, bin_pooling_type,
                            scalar_head_fc_layers, 
                            eval_set,
                            new_analysis,
                            analysis_file_name,
                            model_structure=model_structure,
                            )
    model2_metrics = get_trial_metrics(model2_dir, info_path, n_celltypes, n_filters, bin_size, bin_pooling_type,
                            scalar_head_fc_layers, 
                            eval_set,
                            new_analysis,
                            analysis_file_name,
                            model_structure=model_structure,
                            )
    model1_corr, model2_corr = model1_metrics['scalar_corr'], model2_metrics['scalar_corr']
    if n_trials_to_use is not None:
         model1_corr, model2_corr = model1_corr[:,:n_trials_to_use], model2_corr[:,:n_trials_to_use]
         print('new shape of corr1 and 2', model1_corr.shape, model2_corr.shape)

    idx = identify_regions(model1_corr, model2_corr)
    
    summary_stats(model1_corr, model2_corr, 
                  idx, output_dir, metric_name='Pearson Correlation', label1=label1, label2=label2)
    


# if __name__ == "__main__":
#     main()
def get_lineage_cells(lineage_file, cell_names):
    """
    lineage_file is the first sheet of the excel sheet that can be  downloaded from here: https://ars.els-cdn.com/content/image/1-s2.0-S0092867418316507-mmc1.xlsx
        it is expected to have column names 'CellType' and 'Lineage'
    """
    # Read the lineage file
    df = pd.read_csv(lineage_file)
    
    # Create a mapping of lineages to cell names
    lineage_to_cells = defaultdict(set)
    for _, row in df.iterrows():
        lineage_to_cells[row['Lineage']].add(row['CellType'])

    # Create a mapping of lineages to cell indices
    cell_name_to_index = {name: index for index, name in enumerate(cell_names)}
    lineage_to_indices = {}
    for lineage, cells in lineage_to_cells.items():
        try:
            lineage_to_indices[lineage] = [cell_name_to_index[cell] for cell in cells if cell in cell_name_to_index] # if the cell is not in our list of cell_names don't worry about it
        except ValueError as e:
            missing_cell = str(e).split("'")[1]
            raise ValueError(f"Cell '{missing_cell}' from lineage '{lineage}' not found in cell_names") from e
    
    
    # Make a list of the lineage names (the keys)
    lineage_names = list(lineage_to_indices.keys())

    # Make a list of the lists of cell indices for each lineage
    lineage_cell_indices = list(lineage_to_indices.values())

    return lineage_names, lineage_cell_indices

def get_lineage_counts(lineage_cell_indices, scalar_counts):
    n = len(scalar_counts)
    num_track_groups = len(lineage_cell_indices)
    
    avg_scalar = np.zeros((n, num_track_groups))
    for i, track_group in enumerate(lineage_cell_indices):
        avg_scalar[:, i] = np.mean(scalar_counts[:, track_group], axis=1)
    return avg_scalar


__all__ = ['get_trial_metrics', 'identify_regions', 'summary_stats']
