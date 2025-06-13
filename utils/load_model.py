import torch
import pandas as pd
from torch import nn
from models.BPcm import BPcm
from utils.inference_utils import load_data
from utils.inference_utils import get_least_utilized_gpu
from utils.MemmapDataset import MemmapDataset
from utils.functions import pearson_corr, JSD, spearman_corr
import numpy as np
import os
from models.modules import Bin
from plotting.plot_utils_bpaitac import ocr_mask, get_only_ocrs


seq_len = 998
DEVICE = get_least_utilized_gpu() if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE is", DEVICE)


def load_model(saved_model_path, n_celltypes=90, n_filters=64, bin_size:int = 1, bin_pooling_type:nn.Module=nn.MaxPool1d, scalar_head_fc_layers:int=1, model_structure:nn.Module=None, verbose:bool=True):
  """
    By default loads a BPcm model with the specified number of filters etc.
    If model_structure is used instead, loads the parameters into that model 
    example paths:
    saved_model_path = '/data/nchand/analysis/BP6_L-11/04-16-2023.21.51/best_model'
  """
  print("DEVICE is", DEVICE)
  with torch.no_grad():
    print(seq_len, n_filters, n_celltypes)
    if verbose:
      print("MODEL STRUCTURE", model_structure)
    if model_structure is None:
      model =  BPcm(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
    else:
      model = model_structure
    model.load_state_dict(torch.load(saved_model_path))
    model.to(DEVICE)
    model.eval()
    return model
  
def get_predictions(model:nn.Module, train_loader, val_loader, test_loader, 
                  n_celltypes:int, 
                  get_scalar_prediction:bool, get_profile_prediction:bool,
                  get_scalar_observed:bool, get_profile_observed:bool,
                  eval_set:str,
                  bin_size:int,
                  model_dir:str,
                  saved_file_name:str = 'predictions.npz', 
                  batch_size=100, 
                  save_pred=True,
                  ocr_start:int=375, ocr_end:int=625,
                  seq_len = 998,
                  seq_out_len=998):
  """
  This method takes a model 
  Then it makes model predictions on the eval set 
  Arguments:
    model: the nn Model- that already has the saved model loaded in it
    get_scalar_prediction: if true, will return the scalar predictions
    get_scalar_observed: if true, will return the observed scalar values
    get_profile_prediction: if true, will compute and return the profile predictions
      WARNING: get_profile will SLOW DOWN the running greatly 
    get_profile_observed: if true, will return true profile predictions
  
  Returns: 
    numpy array of scalar_observed, scalar predictions, profile_observed, profile_predictions
    If any of the booleans are set to false, it will return an empty
      array corresponding arrays
  """
  file_path = os.path.join(model_dir, saved_file_name)
  model.to(DEVICE)

  # First check that a file containing all desired data does not already exist
  if os.path.exists(file_path): 
    print(f"Loading existing predictions from {file_path}")
    data = np.load(file_path)
    
    if get_scalar_prediction and not get_profile_prediction:
      data_types = {
        'scalar_obs': get_scalar_observed,
        'scalar_pred': get_scalar_prediction,
      }
    elif get_scalar_prediction and get_profile_prediction:
      data_types = {
        'scalar_obs': get_scalar_observed,
        'scalar_pred': get_scalar_prediction,
        'profile_obs': get_profile_observed,
        'profile_pred': get_profile_prediction
      }
    else: 
      print("WARNING this case has not been fully implemented")
      return 
    
    loaded_data = {
        key: data[key] if key in data and flag and data[key].size > 0 else None
        for key, flag in data_types.items()
    }
    
    if all(loaded_data[key] is not None for key, flag in data_types.items()):
        print("All requested data found in existing file.")
        return tuple(loaded_data.values())
    else:
        print("Some requested data not found in existing file. Recomputing predictions.")


  with torch.no_grad():
    if (eval_set == "validation"):
      loader = val_loader
    elif (eval_set == "training"):
      loader = train_loader
    elif (eval_set == "testing"):
      loader = test_loader
    else:
      print("ERROR unsupported evaluation set:" + eval_set)
      return 0
    
    bin = Bin(width=bin_size, dim=2) # seq len is at dim 2 for profile prediction 
    seq_len = seq_len // bin_size

    print("USING " + eval_set + " data loader")
    # MAKE PREDICTIONS 
    scalar_pred = torch.zeros(0, n_celltypes)
    scalar_obs = torch.zeros(0, n_celltypes)
    profile_pred = torch.zeros(0, n_celltypes, seq_out_len)
    profile_obs = torch.zeros(0, n_celltypes, seq_len)

    for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(loader):
      print("batch num", i)
      seqs, bias, total_counts, bp_counts = seqs.to(DEVICE), seq_bias.to(DEVICE), total_counts.to(DEVICE), bp_counts.to(DEVICE)
      bias = torch.squeeze(seq_bias).to(DEVICE) # the squeezing to work with current head setup
      if bin_size != 1:
          bp_counts = bin(bp_counts).to(DEVICE)
      print('bias shape before it goes into model', bias.size())
      profile, scalar = model(seqs, bias)

      # scalar performance
      if get_scalar_prediction:
        scalar_pred = torch.cat((scalar_pred, scalar.type(torch.FloatTensor)), 0)
      if get_scalar_observed:
        scalar_obs = torch.cat((scalar_obs, total_counts.type(torch.FloatTensor)), 0)
   
      # profile performance note this SLOWS DOWN A LOT
      if get_profile_prediction:
        profile_pred = torch.cat((profile_pred, profile.type(torch.FloatTensor)), 0)
      if get_profile_observed:
        profile_obs = torch.cat((profile_obs, bp_counts.type(torch.FloatTensor)), 0)
        print('jsd mean', torch.mean(JSD(profile_pred, profile_obs[:, :, 375:625])))
    np.savez(file_path, scalar_obs=scalar_obs, scalar_pred=scalar_pred, profile_obs=profile_obs, profile_pred=profile_pred,)
    print('saved', file_path)
    return scalar_obs.numpy(), scalar_pred.numpy(), profile_obs.numpy(), profile_pred.numpy()

def get_scalar_correlation(saved_model_path, n_celltypes=90, info_file_path='info.txt'):
  """ 
  makes predictions from saved_model on validation
  returns a numpy array of correlations between scalar predicted and observed 
  """
  model = load_model(saved_model_path)
  # LOAD THE DATA
  # get dataloaders with the train and validation dat in them
  with torch.no_grad():
    train_loader, val_loader, test_loader = load_data(info_file_path, batch_size=100)
    # train_loader, val_loader = load_data('/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/sample_bias_corrected_normalized_3.7.23/memmap/info.txt', batch_size=100)

    # MAKE PREDICTIONS AND RECORD CORRELATION
    scalar_predictions = torch.zeros(0, n_celltypes)
    scalar_actual = torch.zeros(0, n_celltypes)
    # WE ACTUALLY WANT TO SAVE EACH CORRELATION LIKE IN PLOT_UTILS!
    for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(val_loader):
      seqs, bias, total_counts = seqs.to(DEVICE), seq_bias.to(DEVICE), total_counts.to(DEVICE)
      profile, scalar = model(seqs, bias)
      scalar_predictions = torch.cat((scalar_predictions, scalar.type(torch.FloatTensor)), 0)
      scalar_actual = torch.cat((scalar_actual, scalar.type(torch.FloatTensor)), 0)
      corr = pearson_corr(scalar, total_counts, dim=1) # find correlation for each sequence (take variation accross cells)
      scalar_correlations = torch.cat((scalar_correlations, corr.type(torch.FloatTensor)), 0)
  
    mean_corr = torch.mean(scalar_correlations)
    print('mean correlation', mean_corr)
    return scalar_correlations.numpy()

def get_model(saved_model_path,
               n_celltypes, 
               n_filters,
               bin_size=1, 
               bin_pooling_type='MaxPool1d', 
               scalar_head_fc_layers=1):
  model = load_model(saved_model_path, n_celltypes=n_celltypes, n_filters=n_filters, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  return model

def model_analysis_from_saved_model(saved_model_path:str, n_celltypes:int, n_filters:int, infofile_path:str, 
                  get_scalar_corr:bool, get_profile_corr:bool, get_jsd:bool, 
                  eval_set:str,
                  bin_size:int, bin_pooling_type:nn.Module, scalar_head_fc_layers:int,
                  ocr_start:int=375, ocr_end:int=625, model_structure:nn.Module=None, seq_len=998, 
                  get_complete_corr_metrics=False):
  """
  This method loads the given saved model (assumes BPcm or the model assumed by load_model  with the
  class constant number of celltypes and sequence length)
  If model_structure is provided, then loads that type of model instead, overriding BPcm default
  Then it makes model predictions on the validation dataset 
  and computes the scalar_pearson correlation (w.r.t different celltypes)/ 
  profile base-pair pearson correlation (w.r.t sequence )/JSD of profile
  Arguments:
    saved_model_path file path of model that will be loaded
    infofile_path path to the info.txt file that has information on the 
        location of the memmory mapped data
        ex) '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt'
  """
  model = load_model(saved_model_path, n_celltypes=n_celltypes, n_filters=n_filters, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers, model_structure=model_structure)
  # LOAD THE DATA
  # get dataloaders with the train and validation data in them
  train_loader, val_loader, test_loader = load_data(infofile_path, batch_size=100)
  return model_analysis(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
                  n_celltypes=n_celltypes, 
                  get_scalar_corr=get_scalar_corr, get_profile_corr=get_profile_corr, get_jsd=get_jsd, 
                  eval_set=eval_set,
                  bin_size=bin_size, 
                  ocr_start=ocr_start, ocr_end=ocr_end, seq_len=seq_len, 
                  get_complete_corr_metrics=get_complete_corr_metrics)


def model_analysis(model:nn.Module, train_loader, val_loader, test_loader, 
                  n_celltypes:int, 
                  get_scalar_corr:bool, get_profile_corr:bool, get_jsd:bool, 
                  eval_set:str,
                  bin_size:int, 
                  ocr_start:int=375, ocr_end:int=625, seq_len=998,
                  get_complete_corr_metrics=False):
  """
  This method loads the given model 
  Then it makes model predictions on the validation dataset 
  and computes the scalar_pearson correlation (w.r.t different celltypes)/ 
  profile base-pair pearson correlation (w.r.t sequence )/JSD of profile
  Arguments:
    saved_model_path file path of model that will be loaded
    infofile_path path to the info.txt file that has information on the 
        location of the memmory mapped data
        ex) '/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23/memmap/info.txt'
    get_scalar_corr: if true, will compute and return the scalar pearson corr
    get_pearson_corr: if true, will compute and return the profile pearson corr
    get_jsd: if true, will compute and return the profile jsd
    eval_set: "validation" and "training" and "testing" are supported
  Returns: 
    numpy array of scalar pearson correlation, 
    profile bp pearson corr,
    profile JSD
    ocr profile bp pearson corr
    ocr profile JSD
  If any of the booleans are set to false, it will return an empty
  array corresponding arrays
  """
  with torch.no_grad():
    if (eval_set == "validation"):
      loader = val_loader
    elif (eval_set == "training"):
      loader = train_loader
    elif (eval_set == "testing"):
      loader = test_loader
    else:
      print("ERROR unsupported evaluation set:" + eval_set)
      return 0
    
    bin = Bin(width=bin_size, dim=2) # seq len is at dim 2 for profile prediction 
    seq_len = seq_len // bin_size
    print("USING " + eval_set + " data loader")
    # MAKE PREDICTIONS AND RECORD CORRELATION
    scalar_pred = torch.zeros(0, n_celltypes)
    scalar_obs = torch.zeros(0, n_celltypes)
    scalar_corr = torch.zeros(0)
    scalar_corr_x_ocr = torch.zeros(0)
    spearman_corr_x_celltype = torch.zeros(0)
    spearman_corr_x_ocr = torch.zeros(0)
    profile_corr = torch.zeros(0,n_celltypes)
    OCR_profile_corr = torch.zeros(0,n_celltypes)
    # true_ocr indicates metrics calculated on the data masked to only include the celltype with the most total counts per peak
    top_ocr_profile_corr = torch.zeros(0)
    jsd = torch.zeros(0, n_celltypes)
    OCR_jsd = torch.zeros(0, n_celltypes)
    top_ocr_jsd = torch.zeros(0)
    # WE ACTUALLY WANT TO SAVE EACH CORRELATION LIKE IN PLOT_UTILS!
    for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(loader):
      # make the bp_counts match the size of seq_bias
      if seq_bias.size(-1) < bp_counts.size(-1): # get the middle bp_counts to match the bias seq len
        start = (bp_counts.size(-1) - seq_bias.size(-1)) // 2
        end = start + seq_bias.size(-1)
        bp_counts = bp_counts[:, :, start:end]
        
      print("batch num", i)
      seqs, bias, total_counts, bp_counts = seqs.to(DEVICE), seq_bias.to(DEVICE), total_counts.to(DEVICE), bp_counts.to(DEVICE)
      bias = torch.squeeze(seq_bias).to(DEVICE) # the squeezing to work with current head setup
      if bin_size != 1:
          bp_counts = bin(bp_counts).to(DEVICE)
      profile, scalar = model(seqs, bias)

      # scalar performance
      if get_scalar_corr:
        scalar_pred = torch.cat((scalar_pred, scalar.type(torch.FloatTensor)), 0)
        scalar_obs = torch.cat((scalar_obs, total_counts.type(torch.FloatTensor)), 0)
        s_corr = pearson_corr(scalar, total_counts, dim=1) # find correlation for each sequence (take variation accross cells)
        scalar_corr = torch.cat((scalar_corr, s_corr.type(torch.FloatTensor)), 0)
        if get_complete_corr_metrics:
          spearman_cell = spearman_corr(scalar, total_counts, dim=1)
          spearman_corr_x_celltype = torch.cat((spearman_corr_x_celltype, spearman_cell.type(torch.FloatTensor)), 0)
          corr_x_ocr = pearson_corr(scalar, total_counts, dim=0) #  (take variation accross ocrs)
          scalar_corr_x_ocr = torch.cat((scalar_corr_x_ocr, corr_x_ocr.type(torch.FloatTensor)), 0)
          spearman_ocr = spearman_corr(scalar, total_counts, dim=0)
          spearman_corr_x_ocr = torch.cat((spearman_corr_x_ocr, spearman_ocr.type(torch.FloatTensor)), 0)
      # profile performance
      if get_profile_corr or get_jsd:
        # profile_pred = torch.cat((profile_pred, profile.type(torch.FloatTensor)), 0)
        # bin the profile with summation if bin size is > 1
        # profile_obs = torch.cat((profile_obs, bp_counts.type(torch.FloatTensor)), 0)
        top_ocr_celltype_mask = ocr_mask(total_counts)

        # if the actual profile is full but the predicted profiles is just the OCR region, adjust the actual profile
        if profile.size()[-1] == 250 and bp_counts.size()[-1] > 250:
          # print('Trimming bp counts to be of ocr region only')
          bp_counts = bp_counts[:, :, 375 : 625]

        if get_profile_corr:
          p_corr = pearson_corr(profile, bp_counts, dim=2)
          profile_corr = torch.cat((profile_corr, p_corr.type(torch.FloatTensor)), 0)
          ocr_p_corr = pearson_corr(profile, bp_counts, dim=2)
          OCR_profile_corr = torch.cat((OCR_profile_corr, ocr_p_corr.type(torch.FloatTensor)), 0)
          current_top_ocr_profile_corr = ocr_p_corr[torch.arange(ocr_p_corr.size(0)), top_ocr_celltype_mask]
          top_ocr_profile_corr = torch.cat((top_ocr_profile_corr, current_top_ocr_profile_corr.type(torch.FloatTensor)), 0)
          
        if get_jsd:
          current_jsd = JSD(profile, bp_counts, reduction='none')
          print('mean jsd', torch.mean(current_jsd))
          jsd = torch.cat((jsd, current_jsd.type(torch.FloatTensor)), 0)
          current_ocr_jsd = JSD(profile, bp_counts, reduction='none')
          OCR_jsd = torch.cat((OCR_jsd, current_ocr_jsd.type(torch.FloatTensor)), 0)
          current_top_ocr_jsd = current_ocr_jsd[torch.arange(current_ocr_jsd.size(0)), top_ocr_celltype_mask]
          top_ocr_jsd = torch.cat((top_ocr_jsd, current_top_ocr_jsd.type(torch.FloatTensor)), 0)
        
    if get_complete_corr_metrics:
      corr_x_ocr = pearson_corr(scalar_pred, scalar_obs, dim=0)
      scalar_corr_x_ocr = torch.cat((scalar_corr_x_ocr, corr_x_ocr.type(torch.FloatTensor)), 0)
      spearman_ocr = spearman_corr(scalar_pred, scalar_obs, dim=0)
      spearman_corr_x_ocr = torch.cat((spearman_corr_x_ocr, spearman_ocr.type(torch.FloatTensor)), 0)
    

    mean_s_corr = torch.mean(scalar_corr)
    print('mean scalar correlation', mean_s_corr)
    print('mean scalar spearman corr', torch.mean(spearman_corr_x_celltype))
    if get_complete_corr_metrics:
      return scalar_corr.numpy(), profile_corr.numpy(), jsd.numpy(), OCR_profile_corr.numpy(), OCR_jsd.numpy(), top_ocr_profile_corr.numpy(), top_ocr_jsd.numpy(), scalar_corr_x_ocr.numpy(), spearman_corr_x_celltype.numpy(), spearman_corr_x_ocr.numpy()
    else:
      return scalar_corr.numpy(), profile_corr.numpy(), jsd.numpy(), OCR_profile_corr.numpy(), OCR_jsd.numpy(), top_ocr_profile_corr.numpy(), top_ocr_jsd.numpy()





  



