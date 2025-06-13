import numpy as np
from utils.load_model import get_scalar_correlation, model_analysis, load_data, model_analysis_from_saved_model
import pandas as pd
import os
from datetime import datetime
from scipy.stats import wilcoxon
from torch import nn

N_PEAKS_PER_SET = {'training': 267237, 'validation': 28329, 'testing': 32361}


# This function finds the averages of all the trials from analysis.npz files 

def find_files(directory, filename):
    """Return a list of file paths that match a given filename in a given directory and its subdirectories."""
    file_paths = []
    for root, directories, files in os.walk(directory):
        for file in files:
            if file == filename:
                file_paths.append(os.path.join(root, file))
    return file_paths

def combine_trial_correlations(model_paths, save_path):
    # take a list of paths to models
    # for each path, concatentate the returned correlation
    # save the concatenated correlations? or return them
    combined_corr = np.zeros(0)
    for model in model_paths:
        corrs = get_scalar_correlation(model)
        combined_corr = np.append(corrs, combined_corr)
    np.save(save_path, combined_corr)
    return combined_corr

def combine_trial_metrics(model_paths, val_info_path, n_celltypes:int, n_filters:int, 
                        bin_size:int, bin_pooling_type:nn.Module, scalar_head_fc_layers:int,
                        save_combined:bool, save_trials:bool, save_combined_dir:str, eval_set:str, 
                        only_one_trial:bool=False, analysis_file_name = "analysis.npz",
                        model_structure: nn.Module=None,
                        complete_corr_metrics:bool=False,
                        celltype_corr_metrics:bool=False):
    """
    returns combined scalar correlation, 
    bp correlation, and jsd, OCR bp correlation, OCR JSD,
    and mean for each trial for each.
    eval_set: either "validation" or "training" or "testing"
    saved_combined: when true this saves the means of all metrics across the trials
    save_trials: when this is true it saves the individual trial data for each of the metrics
    """
    if celltype_corr_metrics:
        metric_names = ['scalar_pearson_corr_across_ocrs']
    elif complete_corr_metrics:
        metric_names = ['scalar_corr', 'scalar_spearman_corr_x_celltypes']
    else:
        metric_names = [ 'scalar_corr', 'profile_corr', 'jsd', 'ocr_profile_corr', 'ocr_jsd', 'top_ocr_profile_corr', 'top_ocr_jsd']
    
    
    n_peaks_per_set = {'training': 267237, 'validation': 28329, 'testing': 32361}
    n_peaks = n_peaks_per_set[eval_set]
    combined_metrics = {metric: np.zeros((0, n_celltypes)) for metric in metric_names}

    combined_trial_metrics = {}
    for metric in metric_names:
        if celltype_corr_metrics:
            combined_trial_metrics[metric] = np.zeros((n_celltypes, 0))
        elif metric in ['scalar_corr', 'scalar_spearman_corr_x_celltypes', 'top_ocr_profile_corr', 'top_ocr_jsd']:
            combined_trial_metrics[metric] = np.zeros((n_peaks, 0))
        else:
            combined_trial_metrics[metric] = np.zeros((n_peaks, n_celltypes, 0))

    trial_means = {metric: [] for metric in metric_names}

    # analysis_file_name = "analysis.npz" if eval_set == "validation" else "analysis_" + eval_set + ".npz" 
    print("Analysis file name is", analysis_file_name)
    for model in model_paths:
        directory = os.path.dirname(model)
        # check if the analysis metrics have already been calculated:
        # THIS ONLY worKS IF PAST ANALYSIS ALSO HAD ocr_jsd and ocr_bp_corr
        if (os.path.isfile(os.path.join(directory, analysis_file_name))):
            print("using existing", os.path.join(directory, analysis_file_name))
            analysis = np.load(os.path.join(directory, analysis_file_name))
            metrics = {metric: analysis[metric] for metric in metric_names if metric in analysis}

        else : # calculate the metrics from scratch 
            metrics_tuple = model_analysis_from_saved_model(
                model, n_celltypes, n_filters, val_info_path, 
                get_scalar_corr='scalar_corr' in metric_names, 
                get_profile_corr='profile_corr' in metric_names, 
                get_jsd='jsd' in metric_names, 
                eval_set=eval_set, 
                bin_size=bin_size, bin_pooling_type=bin_pooling_type, 
                scalar_head_fc_layers=scalar_head_fc_layers,
                model_structure=model_structure)
            metrics = dict(zip(metric_names, metrics_tuple))
            np.savez(os.path.join(directory, analysis_file_name), **metrics)
            print("saved analysis file here", os.path.join(directory, analysis_file_name))

        for metric in metric_names:
            if metric in metrics:
                trial_means[metric].append(np.mean(metrics[metric]))
                if metrics[metric].ndim == 1:
                    combined_metrics[metric] = np.append(combined_metrics[metric], metrics[metric])
                    # if save_trials:
                    #     combined_trial_metrics[metric] = np.concatenate(combined_trial_metrics[metric], metrics[metric], axis=1)
                else:
                    combined_metrics[metric] = np.append(combined_metrics[metric], metrics[metric], axis=0)
                    # if save_trials:
                    #     combined_trial_metrics[metric] = np.concatenate(combined_trial_metrics[metric], metrics[metric], axis=2)
                if save_trials:
                    print(combined_trial_metrics[metric].shape)
                    print('shape metrics[metric]', metrics[metric])
                    combined_trial_metrics[metric] = np.concatenate((combined_trial_metrics[metric], np.expand_dims(metrics[metric], axis=-1)), axis=-1)
                    # combined_trial_metrics[metric] = np.concatenate((combined_trial_metrics[metric], metrics[metric]), axis=-1)
        if only_one_trial:
            break
    
    if save_combined:
        print("saving combined trial data in", os.path.join(save_combined_dir, 'analysis_' + eval_set + '_trials.npz'))
        np.savez(os.path.join(save_combined_dir, 'analysis_' + eval_set + '_trials_combined.npz'), **combined_metrics, **{f"{metric}_trial_mean": trial_means[metric] for metric in metric_names})

    if save_trials:
        print("saving individual trial data in", os.path.join(save_combined_dir, 'analysis_' + eval_set + '_trials.npz'))
        np.savez(os.path.join(save_combined_dir, 'analysis_' + eval_set + '_trials.npz'), **combined_trial_metrics)

    return combined_metrics, trial_means
    # return scalar_combined, sc_trial_mean, bp_corr_combined, bp_corr_trial_mean, jsd_combined, jsd_trial_mean, ocr_bp_corr_combined, ocr_bp_corr_trial_mean, ocr_jsd_combined, ocr_jsd_trial_mean


def combine_multiple_models(results_dir, lambda_dirs, lambdas, val_info_path, n_celltypes, n_filters, bin_sizes, bin_pooling_type, scalar_head_fc_layers, eval_set, experiment_names, analysis_file_name, save_trials=False, max_n_trials=None, complete_corr_metrics=False, celltype_corr_metrics=False):
    """
    Designed to take in a list of differnet lambdas in separate folders
    Finds all models saved within each of the labmda folders, and combines
    all the trials from that model into a dataframe
    saved_trials: boolean indicating whether a file containing all of the data separated into different rows for each trial should be saved 
        the single dataframe with all data concatenated will be saved already automatically. 

    returns: 
        1) a dataframe containing the scalar correlation, for each OCR
        mean base pair correlation across all celltypes for each OCR
        and mean base pair profile JSD across all celltypes
        2) a dataframe containing the means of each metric for each of the trials 
        found for each model.
    
    """    
    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    analysis_df = pd.DataFrame()
    trial_df = pd.DataFrame()

    # for each lambda directory to be included in the plot
    # use find_files to find all the best_model files of a certain lambda
    for i in range(len(lambda_dirs)):
        lambda_dir = lambda_dirs[i]
        print("current dir:", lambda_dir)
        cur_bin_size = bin_sizes[i]
        paths = find_files(os.path.join(results_dir, lambda_dir), 'best_model')
        print("paths found", paths, "in", os.path.join(results_dir, lambda_dir))
        if max_n_trials is not None:
            paths = paths[:max_n_trials]
            print("REDUCED N TRIAL DATA POINTS TO", max_n_trials)

        # get the correlations for all trials of a given lambda using combine_trial_..()
        combined_metrics, trial_means = combine_trial_metrics(paths, val_info_path, n_celltypes, n_filters, bin_size=cur_bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers, save_combined=save_trials, save_trials=save_trials, save_combined_dir=os.path.join(results_dir, lambda_dir), eval_set=eval_set, only_one_trial=False, analysis_file_name=analysis_file_name,complete_corr_metrics=complete_corr_metrics, celltype_corr_metrics=celltype_corr_metrics)

        if celltype_corr_metrics:
            shape_of_lambda = combined_metrics['scalar_pearson_corr_across_ocrs'].shape
        else:
            shape_of_lambda = combined_metrics['scalar_corr'].shape
        lambda_array = np.full(shape=shape_of_lambda, fill_value=lambdas[i])
        print('lambda is ', lambdas[i])
        
        if celltype_corr_metrics:
            sc_current_df = pd.DataFrame({'scalar_pearson_corr_across_ocrs': np.array(combined_metrics['scalar_pearson_corr_across_ocrs']), 
                                      'lambda': np.array(lambda_array), 
                                      'experiment_name': np.full(shape=combined_metrics['scalar_pearson_corr_across_ocrs'].shape, fill_value=experiment_names[i])})
            print('len sc_current_df a single lambda', len(sc_current_df))
            analysis_df = analysis_df.append(sc_current_df)
            print(f'len analysis_df at step {i}: {len(analysis_df)}')
    
            # record mean from all trials in df so we can plot them
            trials = np.arange(len(trial_means['scalar_pearson_corr_across_ocrs'])) + 1
            trials_current_df = pd.DataFrame({'scalar_pearson_corr_across_ocrs': trial_means['scalar_pearson_corr_across_ocrs'], 
                                            'lambda': np.full(shape=len(trial_means['scalar_pearson_corr_across_ocrs']), fill_value=lambdas[i]), 'trial': trials, 
                                            'experiment_name': np.full(shape=len(trial_means['scalar_pearson_corr_across_ocrs']), fill_value=experiment_names[i])})
            trial_df = trial_df.append(trials_current_df)

        elif analysis_file_name == 'testing_correlation_analysis.npz':
            sc_current_df = pd.DataFrame({'scalar_corr': np.array(combined_metrics['scalar_corr']), 
                                      'scalar_spearman_corr_x_celltypes': np.array(combined_metrics['scalar_spearman_corr_x_celltypes']),
                                      'lambda': np.array(lambda_array), 
                                      'experiment_name': np.full(shape=combined_metrics['scalar_corr'].shape, fill_value=experiment_names[i])})
            print('len sc_current_df a single lambda', len(sc_current_df))
            analysis_df = analysis_df.append(sc_current_df)
            print(f'len analysis_df at step {i}: {len(analysis_df)}')
    
            # record mean from all trials in df so we can plot them
            trials = np.arange(len(trial_means['scalar_corr'])) + 1
            trials_current_df = pd.DataFrame({'scalar_corr': trial_means['scalar_corr'], 
                                              'scalar_spearman_corr_x_celltypes': trial_means['scalar_spearman_corr_x_celltypes'],
                                            'lambda': np.full(shape=len(trial_means['scalar_corr']), fill_value=lambdas[i]), 'trial': trials, 
                                            'experiment_name': np.full(shape=len(trial_means['scalar_corr']), fill_value=experiment_names[i])})
            trial_df = trial_df.append(trials_current_df)
        
        else:   
            # now we average bp_corr and jsd across celltypes
            bp_corr_mean = np.mean(combined_metrics['profile_corr'], -1)

            jsd_mean = np.mean(combined_metrics['jsd'], axis=1)
            ocr_bp_corr_mean = np.mean(combined_metrics['ocr_profile_corr'], -1)
            ocr_jsd_mean = np.mean(combined_metrics['ocr_jsd'], axis=1)
            # We do not take the mean of top_ocr metrics across celltypes because they only represent one celltype
            # the top ocr metrics only have one celltype so we will not mean them
            sc_current_df = pd.DataFrame({'scalar_corr': np.array(combined_metrics['scalar_corr']), 
                                        'bp_corr_mean': np.array(bp_corr_mean), 
                                        'jsd_mean': np.array(jsd_mean), 
                                        'ocr_bp_corr_mean': np.array(ocr_bp_corr_mean), 
                                        'ocr_jsd_mean': np.array(ocr_jsd_mean), 
                                        'top_ocr_bp_corr': np.array(combined_metrics['top_ocr_profile_corr']), 
                                        'top_ocr_jsd': np.array(combined_metrics['top_ocr_jsd']), 
                                        'lambda': np.array(lambda_array), 
                                        'experiment_name': np.full(shape=combined_metrics['scalar_corr'].shape, fill_value=experiment_names[i])})
            print('len sc_current_df a single lambda', len(sc_current_df))
            analysis_df = analysis_df.append(sc_current_df)
            print(f'len analysis_df at step {i}: {len(analysis_df)}')

            # record mean from all trials in df so we can plot them
            trials = np.arange(len(trial_means['scalar_corr'])) + 1
            trials_current_df = pd.DataFrame({'scalar_corr': trial_means['scalar_corr'], 
                                            'bp_corr_mean': trial_means['profile_corr'], 
                                            'jsd_mean': trial_means['jsd'], 
                                            'ocr_bp_corr_mean': trial_means['ocr_profile_corr'], 
                                            'ocr_jsd_mean': trial_means['ocr_jsd'], 
                                            'top_ocr_bp_corr_mean': trial_means['top_ocr_profile_corr'], 
                                            'top_ocr_jsd_mean': trial_means['top_ocr_jsd'], 
                                            'lambda': np.full(shape=len(trial_means['scalar_corr']), fill_value=lambdas[i]), 'trial': trials, 
                                            'experiment_name': np.full(shape=len(trial_means['scalar_corr']), fill_value=experiment_names[i])})
            trial_df = trial_df.append(trials_current_df)

    return analysis_df, trial_df


# params: files: a list of filepaths to analysis.npz files that are all trials 
#           of the same experiment which need to be combined. Note: these experiments
#           must have outputs corresponding to peaks/regions in the SAME order
#       metric: the name of the file withing the analysis file that you would like to average
#           ex) 'scalar_corr'
# Outputs a txt file with the following columns: peak names, 
# and mean averages for all of the given metric in the input npzs accross the given trials
def avg_across_trials(analysis_files, metric):
    # assume that all analysis_files have the same numpy files within them

    # make an output numpy ndarray of zeros of size num_peaks x number of metrics
    n_peaks = np.load(analysis_files[0])[metric].shape[0]
    print("num peaks", n_peaks)
    n_trials = len(analysis_files)
    print("num trials", n_trials)
    sum = np.zeros((n_peaks))

    # get the sum over all trials
    for filepath in analysis_files:
        analysis = np.load(filepath)
        data = analysis[metric]
        sum += data
        print(sum[:5])

    # divide by the number of trials
    mean = sum / n_trials
    return mean
    
# This method finds all trials (aka analysis files) in a directory and
# averages a scalar corr, and then returns a txt file with two columns:
# the peak names and the peak scalar correlation
# params: 
#   dir: directory that will be searched for analysis files. ALL analysis.npz
#       files within this directory will be treated like separate trials and averaged
#   peak_names: path to the numpy file containing the names of all the peaks
def get_scalar_corr_avg_from_files(dir, peak_names, output_path):
    # get the analysis files
    analysis_files = find_files(dir, 'analysis.npz')
    # get the averages
    averages = avg_across_trials(analysis_files, 'scalar_corr')
    out = np.vstack((peak_names, averages))
    out = np.transpose(out)
    print("combined shape", out.shape)
    np.savetxt(output_path, out, fmt='%s', delimiter='\t')

# returns a pandas dataframe with a column of peak names
# and a column of avg scalar_corr accross all of the analysis files 
# contained in each model_name directory
# note: experiment_vars used to be called lambdas
def get_combined_avg(peak_names, model_names, experiment_vars, results_dir, analysis_filename='analysis.npz'):
    combined_avgs = pd.DataFrame()
    combined_avgs['peak_names'] = peak_names
    for i in range(len(model_names)):
        model = model_names[i]
        var = experiment_vars[i]
        dir = results_dir + model
        analysis_files = find_files(dir, analysis_filename)
        print('found files', analysis_files)
        combined_avgs[str(var)] = avg_across_trials(analysis_files, 'scalar_corr')
    return combined_avgs

# this function computes the Wilcoxon Rank Sum Test between
# the second column (idx 1) of the given data file (assumed to be lambda 0)
# and all following columns of the input data
# params: data_path: a tsv file containing a metric for each model in columns 1: 
#   where the first column is the lambda=0 model or another model that all others
#   will be compared against
def wilcoxon_rank_sum_test_between_models(data_path):
    data = pd.read_csv(data_path, sep='\t')
    other_model_names = data.columns[2:]  # Assuming the first column is not a model name
    print("model names", other_model_names)
    model1_data = data[data.columns[1]]  # First model's data
    results = []

    # Iterate through the model names and perform Wilcoxon rank sum test
    for model_name in other_model_names:
        model2_data = data[model_name]  # Data for the current model
        stat, p_value = wilcoxon(model1_data, model2_data)
        results.append({
            'Model1': data.columns[1],
            'Model2': model_name,
            'Statistic': stat,
            'P-value': p_value
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df



def main():
    # For BP17 run with complete_bias_corrected_normalized_4.3.23-2 (on Hyak) ONLY!
    #  Peak names may be different for other datasets
    # peak_names = np.load('/data/nchand/analysis/BPcm/BP17_setup_data/val_peak_names.npy') 
    # results_dir = '/data/nchand/analysis/BPcm/'
    # model_names = ['BP17_L0_0', 'BP17_L-1_1', 'BP17_L-1_3', 'BP17_L-1_5', 'BP17_L-1_7', 'BP17_L-1_9',
    #           'BP17_L0_1.1', 'BP17_L0_1.3', 'BP17_L0_1.5']
    # lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9,
    #          1.1, 1.3, 1.5]
    
    # For BP33 run with complete_bias_corrected_normalized_4.3.23-2 (on Hyak)
    peak_names = np.load('/data/nchand/analysis/BPcm/BP17_setup_data/val_peak_names.npy') # i think it was the same as BP17
    results_dir = '/data/nchand/analysis/BPcm/'
    model_names = ['BP33_1_L0_0.9', 'BP33_5_L0_0.9']
    experiment_variables = [1, 5]
    
    # Getting averages
    avgs = get_combined_avg(peak_names, model_names, experiment_variables, results_dir)
    print(avgs.shape)
    print(avgs)
    # current_date = datetime.now().strftime('%Y-%m-%d')
    # out_path = results_dir + 'scalar_corr_avgs' + current_date + '.tsv'
    # avgs.to_csv(out_path, sep='\t', index=False)
    # print("saved avgs from given models at", out_path)

    # # save txt files for specific lambdas to run in Alex's scatter plot scripts
    # lambda_0 = avgs[['peak_names', '0']]
    # lambda_0.to_csv('/data/nchand/analysis/BPcm/BP17_L0_0/scalar_corr_avg.tsv', sep='\t', index=False)
    # lambda_p7 = avgs[['peak_names', '0.7']]
    # lambda_p7.to_csv('/data/nchand/analysis/BPcm/BP17_L-1_7/scalar_corr_avg.tsv', sep='\t', index=False)
    # lambda_p9 = avgs[['peak_names', '0.9']]
    # lambda_p9.to_csv('/data/nchand/analysis/BPcm/BP17_L-1_9/scalar_corr_avg.tsv', sep='\t', index=False)

    bin1 = avgs[['peak_names', '1']]
    bin1.to_csv('/data/nchand/analysis/BPcm/BP33_1_L0_0.9/scalar_corr_avg.tsv', sep='\t', index=False, header=False)
    bin5 = avgs[['peak_names', '5']]
    bin5.to_csv('/data/nchand/analysis/BPcm/BP33_5_L0_0.9/scalar_corr_avg.tsv', sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
