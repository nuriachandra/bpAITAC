import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
import subprocess
import argparse
from typing import List # use to specify list type
from typing import Optional
import numpy as np
from typing import Tuple, Dict
from utils.inference_utils import load_data_non_memmaped
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
from models.BPnetRep import BPnetRep
from models.BPmultimax import BPmultimax
from models.BPcm import BPcm
from models.bpAITAC import bpAITAC
from models.BPcm_250 import BPcm_250
from models.BPbi import BPbi
from models.BPbi_shallow import BPbi_shallow
from models.BPcm_bias0 import BPcm_bias0
from models.BPcm_biasRandom import BPcm_biasRandom
from models.BPcm_noOffTwo import BPcm_noOffTwo
from models.BPoh import BPoh
from models.BPol import BPol
from models.BPmp import BPmp
from models.BPcm_skinny import BPcm_skinny
from models.BPcm_super_skinny import BPcm_super_skinny
from utils.EarlyStopping import EarlyCorrelationStopping, EarlyStopping
import time
import copy
import datetime
import sys
import json
from plotting.plot_results import plot_training_val_loss
from  utils.functions import JSD, pearson_corr, ocr_pearson_corr
from  utils.load_model import model_analysis
from  utils.MemmapDataset import MemmapDataset 
import pandas as pd
from  models.modules import CompositeLoss, ProfileBasedLoss, CompositeLossBalanced, CompositeLossBalancedJSD, CompositeLossMNLL, Bin, CompositeLossMSE, PoissonLoss, MSEDoubleBP, MSEDoubleProfile
from utils.LogResults import LogResults
from typing import Literal
from plotting.plot_utils_bpaitac import ocr_mask

LRSchedulerType = Literal[
    "StepLR",
    "CosineAnnealingLR",
    "WarmupCosineDecayLR",  
]

def load_data(info_file:str, batch_size=100):
  """
  Loads data from memmaps into the MemmapDataset,
  and returns DataLoaders containing the MemmapDatasets for training set and validation set

  info_file: the path to the "info.txt" file in the a folder of memmap data produced by prep_data.py
      info.txt contains file paths and datatypes and shapes for all the memmorymapped data 
      info_file for memmaped data format:
      1st column is memmap file path, 2nd column is dtype, 3rd col is shape
      example:
      '/gscratch/mostafavilab/nchand/bpAITAC/data_train_test/complete/memmap/test.names.dat'	dtype('<U26')	(267237,)
  batch_size: the batch size to be used in each training iteration. 
      defaults to 100
  """
  # load the file as a csv
  info_df = pd.read_csv(info_file, delimiter="\t", names=['path', 'dtype', 'shape'])

  info_df['path'] = info_df['path'].apply(lambda x: x[1:-1]) # git rid of quotation marks in the info_file strings
  info_df['shape'] = info_df['shape'].apply(lambda x: x[1:-1]) # rid of parentheses in 'shapes'
 

  train_df = info_df.iloc[1:5]
  # train_df = info_df.iloc[6:10]
  val_df = info_df.iloc[11:15]
  dataframes = [train_df, val_df]
  train_loader:torch.utils.data.DataLoader = None
  val_loader:torch.utils.data.DataLoader = None
  dataloader = [train_loader, val_loader]

  for j in range(2): # val and then train
    df = dataframes[j]
    file_types = [] # [bp_counts, total_counts, bias, onehot]
    # iterate through all data files that need to be included in data loader
    for i in range(4): 
      shape = tuple(int(num) for num in df.loc[df.index[i]].at['shape'].split(", ")) # parsing the string representation of the shape
      file_types.append(np.memmap(df.iloc[i,0], dtype='float32', shape=shape)) # loading memory map 

    dataset = MemmapDataset(file_types[3], file_types[0], file_types[1], file_types[2]) # [bp_counts, total_counts, bias, onehot]
    dataloader[j] = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

  return dataloader[0], dataloader[1]

def trim_to_center(x, len, off_by_two):
    last_dim_size = x.size(-1)
    if off_by_two:
      last_dim_size += 2
    start_idx = (last_dim_size - len) // 2
    end_idx = start_idx + len
    out = x[..., start_idx:end_idx]
    return out

def train(
    model: nn.Module,
    train_loader: DataLoader, val_loader: DataLoader,
    loss_fxn_name:str,
    output_directory: str,
    epochs: int,
    _lambda:int,
    bias_model:bool,
    ocr_eval:bool,
    bin_size:int,
    seq_len:int=1000,
    ocr_start:int = 375,
    ocr_end:int = 625,
    learning_rate:int = 0.001, 
    scheduler_name: Optional[LRSchedulerType] = None,
    scheduler_kwargs: dict = None,
    trainvariability_cut: float = 0.2, # reduce LR if training loss increases by this percent of the initial loss
    trainvariability_cutn: int = 3, # reduce LR if training loss increases for this many epochs - will make training take longer
    save_best_loss_model: bool = False,
    off_by_two: bool = True,
    patience:int = 10,
    ):
  """
  Trains the given model with Poisson negative log likelihood loss
  and Adam optimizer 
  returns a model loaded with the best state from validation
    and the value of the best validation loss
  
  model: an nn.Module that will be trained
  train_loader: a DataLoader containing the onehot-encoded input data and 
      labels (the number of ATAC-seq reads) to be used for training
  train_loader: a DataLoader containing the onehot-encoded input data and 
      labels (the number of ATAC-seq reads) to be used for validation
  output_directory: a file path where the loss per epoch will be stored 
  epochs: the number of epochs you want to train the model for
  _lambda: controls the weight that profile error is given 
      (where total count prediciton error automatically has weight 1)
      only used when loss function is composite
  ocr_eval: bool controls whether the profile is evaluated on its entire loss
      or just the 250 bp region from 375 to 625. Only relevant 
      when loss function is CompositeLoss
  bias_model: bool: if true, then bias will be added when learning. otherwise bias will be set
      to zero, which translates to there being no bias added
  bin_size: the number of bps that will be "binned" in the profile
  learning_rate: is the starting learning rate if using a learning rate scheduler 
  """
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  DEVICE = get_least_utilized_gpu()
  print(DEVICE)  # this should print out CUDA 
  model.to(DEVICE)

  if hasattr(model, "profile_out_len"):
    profile_out_len = model.profile_out_len
  else:
    profile_out_len = seq_len
    

  
  bin = Bin(width=bin_size, dim=2) # seq len is at dim 2 for profile prediction
  early_loss_stopping = EarlyStopping(patience=patience, delta=0)
  early_corr_stopping = EarlyCorrelationStopping(patience=patience, delta=0.0)

  if loss_fxn_name == "PoissonNLLLoss":
    loss_criterion = nn.PoissonNLLLoss(log_input=False)
  elif loss_fxn_name == "CompositeLoss":
    loss_criterion = CompositeLoss(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "CompositeLossBalanced":
    loss_criterion = CompositeLossBalanced(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "CompositeLossBalancedJSD":
    loss_criterion = CompositeLossBalancedJSD(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "CompositeLossMNLL":
    loss_criterion = CompositeLossMNLL(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "ProfileBasedLoss":
    loss_criterion = ProfileBasedLoss(ocr_start, ocr_end)
  # elif loss_fxn_name == "MSELoss":
  #   loss_criterion = nn.MSELoss(reduction='mean')
  elif loss_fxn_name == "CompositeLossMSE":
    loss_criterion = CompositeLossMSE(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "PoissonLoss":
    loss_criterion = PoissonLoss(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "MSEDoubleBP":
    loss_criterion = MSEDoubleBP(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  elif loss_fxn_name == "MSEDoubleProfile":
    loss_criterion = MSEDoubleProfile(reduction='mean', ocr_only=ocr_eval, ocr_start=ocr_start, ocr_end=ocr_end)
  else:
    print("loss function not yet supported")
    exit()

  print("loss criterion:", loss_criterion)
  
  optimizer = torch.optim.Adam(model.parameters(), learning_rate)
  
  # set up learning rate scheduler (if provided)
  scheduler = None
  if scheduler_name is not None and not "None":
    if scheduler_kwargs is None:
      scheduler_kwargs = {}
    scheduler = get_learning_rate_scheduler(scheduler_name, optimizer, **scheduler_kwargs)


  total_step = len(train_loader)
  # set up classes that will record results
  train_logs = LogResults("train", output_directory, loss_fxn_name)
  val_logs = LogResults("val", output_directory, loss_fxn_name)


  best_loss_model_wts = copy.deepcopy(model.state_dict())
  best_corr_model_wts = copy.deepcopy(model.state_dict())
  best_loss_val = float('inf')
  best_corr_val = 0.0
  best_epoch = 1

  eps = 1e-8

  train_loss_before = float('inf')
  train_loss_increased = 0
  beginning_loss = float('inf')

  for epoch in range(epochs):
      running_loss = 0.0
      running_jsd = 0.0
      running_ocr_jsd = 0.0
      running_top_ocr_jsd = 0.0  
      running_corr = 0.0
      running_bp_corr = 0.0
      running_ocr_bp_corr = 0.0
      running_scalar_loss = 0.0
      running_profile_loss = 0.0

      model.train()
      for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(train_loader):  
          if profile_out_len < bp_counts.size(-1):
            bp_counts = trim_to_center(x=bp_counts, len=profile_out_len, off_by_two=off_by_two)
          if profile_out_len < seq_bias.size(-1):
            seq_bias = trim_to_center(x=seq_bias, len=profile_out_len, off_by_two=off_by_two)

          seqs = seqs.to(DEVICE) # convert numpy to torch and send to GPU
          # We transpose the one-hot encoded data to make it (4, seq_len), (4 from 4 nucleotides)
          # because then it will work with 1d convolutions better
          bp_counts = bp_counts.to(DEVICE) # labels are the real ATAC-seq peak heights for 81 cell types (81 columns)
          if bin_size != 1:
            bp_counts = bin(bp_counts).to(DEVICE)

          total_counts = total_counts.to(DEVICE)
          if bias_model:
            bias = torch.squeeze(seq_bias).to(DEVICE) # the squeezing to work with current head setup
          else: 
            bias = torch.from_numpy(np.zeros(shape=(seq_bias.size(dim=0), seq_len), dtype=np.double)).to(DEVICE)
            # we need to convert this to a float 32
            bias = bias.float()

          profile, scalar = model(seqs, bias)

          # FOR DEBUGGING
          if torch.isnan(torch.mean(profile)):
            print("IN TRAIN THE PROFILE HAS AN NAN")
            print(profile)
            print(profile.size())

          if loss_fxn_name == "CompositeLoss" or "CompositeLossBalanced" or "CompositeLossBalancedJSD" or "CompositeLossMNLL": # this just always evals to true lol. should fix in the future
            #  bp_counts, total_counts, profile_prediciton, total_count_prediction, _lambda
            loss, scalar_loss, profile_loss = loss_criterion(bp_counts, total_counts, profile, scalar, _lambda)
            running_scalar_loss += scalar_loss.item()
            running_profile_loss += profile_loss.item()
          elif loss_fxn_name == "ProfileBasedLoss":
            # note: in this case profile must not be a probability distribution
            # must be a prediction of the straight up profile 
            loss, scalar_loss, profile_loss = loss_criterion(bp_counts, total_counts, profile, _lambda)
            running_scalar_loss += scalar_loss.item()
            running_profile_loss += profile_loss.item()
          else: # loss fxn is MSELoss or PoissonLoss 
            # unsqueeze the scalar output, (add dim of 1 to end) so that it can be broadcast across the sequence length
            scalar_head_vector = torch.unsqueeze(scalar, scalar.dim())
            # element-wise multiply the predicted total counts for each cell type by the profile
            prediction = torch.mul(profile, scalar_head_vector)
            loss = loss_criterion(prediction, bp_counts)

          running_loss += loss.item()

          # keep track of sum of pearson correlation between predicted total counts and real total counts
          running_corr += torch.sum(pearson_corr(scalar, total_counts, dim=1)).item()
          running_jsd += torch.sum(torch.mean(JSD(profile, bp_counts, reduction='none'), dim=1)).item() # take mean across celltypes for each JSD
          if bin_size == 1: # only look at ocr specific thing when sequence is not binned
            current_ocr_jsd = JSD(profile[:, :, ocr_start:ocr_end].to(DEVICE), bp_counts[:, :, ocr_start:ocr_end].to(DEVICE), reduction='none')
            running_ocr_jsd += torch.sum(torch.mean(current_ocr_jsd, dim=1)).item()
            top_ocr_celltype_mask = ocr_mask(total_counts) 
            running_top_ocr_jsd += torch.sum(current_ocr_jsd[torch.arange(current_ocr_jsd.size(0)), top_ocr_celltype_mask]).item()

          # start training model with backward and optimize only after the zeroth epoch
          if (epoch != 0):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # update LR scheduler 
            if scheduler is not None:
              scheduler.step()
              
          if (i) % 100 == 0:
              print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Profile Loss: {:.4f}, Scalar Loss: {:.4f}'
                    .format(epoch, epochs, i, total_step, loss.item(), profile_loss.item(), scalar_loss.item()))
      num_batches = i + 1
      if loss_fxn_name == 'CompositeLoss' or 'CompositeLossBalanced' or 'CompositeLossBalancedJSD' or "CompositeLossMNLL":
        train_logs.record_results(epoch, len(train_loader.dataset), num_batches,
              running_loss, running_jsd, running_corr, running_bp_corr, running_ocr_jsd, running_ocr_bp_corr, running_top_ocr_jsd, 
              running_scalar_loss, running_profile_loss)
      
      avg_epoch_loss = running_loss / len(train_loader.dataset)
      if epoch == 0:
        beginning_loss = avg_epoch_loss
        train_loss_before = beginning_loss
      
      if (epoch > 0) and ((avg_epoch_loss - train_loss_before) / beginning_loss > trainvariability_cut or train_loss_increased >= trainvariability_cutn):
        if (avg_epoch_loss - train_loss_before) / beginning_loss > trainvariability_cut:
          print("(avg_epoch_loss - train_loss_before) / beginning_loss > trainvariability_cut")
        if (train_loss_increased >= trainvariability_cutn):
          print("train_loss_increased >= trainvariability_cutn")
        for param_group in optimizer.param_groups:
          param_group['lr'] *= 0.1
        print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']} at epoch {epoch+1}")
        train_loss_increased = 0
      
      if avg_epoch_loss > train_loss_before:
        train_loss_increased += 1
      else:
        train_loss_increased = 0
      train_loss_before = avg_epoch_loss
      
      # calculate val loss for epoch
      running_val_loss = 0.0
      running_jsd = 0.0
      running_ocr_jsd = 0.0
      running_top_ocr_jsd = 0.0
      running_corr = 0.0
      running_bp_corr = 0.0
      running_ocr_bp_corr = 0.0
      running_scalar_loss = 0.0
      running_profile_loss = 0.0
      with torch.no_grad():
          model.eval()
          for i, (seqs, bp_counts, total_counts, seq_bias) in enumerate(val_loader):
              # if seq_bias.size(-1) < bp_counts.size(-1): # get the middle bp_counts to match the bias seq len
              #   start = (bp_counts.size(-1) - seq_bias.size(-1)) // 2
              #   end = start + seq_bias.size(-1)
              #   bp_counts = bp_counts[:, :, start:end]
              if profile_out_len < bp_counts.size(-1):
                bp_counts = trim_to_center(x=bp_counts, len=profile_out_len, off_by_two=off_by_two)
              if profile_out_len < seq_bias.size(-1):
                seq_bias = trim_to_center(x=seq_bias, len=profile_out_len, off_by_two=off_by_two)

              x = seqs.to(DEVICE)
              y = bp_counts.to(DEVICE)
              if bin_size != 1:
                y = bin(bp_counts).to(DEVICE)

              total_counts = total_counts.to(DEVICE)
              if bias_model:
                bias = torch.squeeze(seq_bias).to(DEVICE) # the squeezing to work with current head setup
              else: bias = torch.from_numpy(np.zeros(shape=(seq_bias.size(dim=0), seq_len))).to(DEVICE)
              profile, scalar = model(x, bias)
              # FOR DEBUGGING
              if torch.isnan(torch.mean(profile)):
                print(profile)
    
              if loss_fxn_name == "CompositeLoss" or "CompositeLossBalanced" or 'CompositeLossBalancedJSD' or "CompositeLossMNLL":
                loss, scalar_loss, profile_loss = loss_criterion(y, total_counts, profile, scalar, _lambda)
                running_scalar_loss += scalar_loss.item()
                running_profile_loss += profile_loss.item()
              elif loss_fxn_name == "ProfileBasedLoss":
                # note: in this case profile must not be a probability distribution
                # must be a prediction of the straight up profile 
                # TODO add ocr loss if ever use this
                loss, scalar_loss, profile_loss = loss_criterion(y, total_counts, profile, _lambda)
                running_scalar_loss += scalar_loss.item()
                running_profile_loss += profile_loss.item()
              else:
                scalar_head_vector = torch.unsqueeze(scalar, scalar.dim())
                prediction = torch.mul(profile, scalar_head_vector) 
                loss = loss_criterion(prediction, y)

              running_val_loss += loss.item()
              running_jsd += torch.sum(torch.mean(JSD(profile, y, reduction='none'), dim=1)).item() # mean jsd across celltypes
              if bin_size == 1: # only look at ocr specific thing when sequence is not binned
                current_ocr_jsd = JSD(profile[:, :, ocr_start:ocr_end].to(DEVICE), bp_counts[:, :, ocr_start:ocr_end].to(DEVICE), reduction='none')
                running_ocr_jsd += torch.sum(torch.mean(current_ocr_jsd, dim=1)).item()
                top_ocr_celltype_mask = ocr_mask(total_counts) 
                running_top_ocr_jsd += torch.sum(current_ocr_jsd[torch.arange(current_ocr_jsd.size(0)), top_ocr_celltype_mask]).item()

              running_corr += torch.sum(pearson_corr(scalar, total_counts, dim=1)).item()
              # running_bp_corr += torch.sum(torch.mean(pearson_corr(profile, y, dim=2), dim=1)).item() 
              # running_ocr_bp_corr += torch.sum(torch.mean(pearson_corr(profile[:,:,ocr_start:ocr_end], y[:,:,ocr_start:ocr_end], dim=2), dim=1)).item() 

      num_batches = i + 1
      if loss_fxn_name == 'CompositeLoss' or "CompositeLossBalanced" or 'CompositeLossBalancedJSD' or "CompositeLossMNLL":
        val_logs.record_results(epoch=epoch, dataset_len=len(val_loader.dataset), n_batches=num_batches,
          loss=running_val_loss, jsd=running_jsd, correlation=running_corr, bp_correlation=running_bp_corr, 
          ocr_jsd=running_ocr_jsd, ocr_bp_correlation=running_ocr_bp_corr, top_ocr_jsd=running_top_ocr_jsd,
          scalar_error=running_scalar_loss, profile_error=running_profile_loss)

      val_loss = running_val_loss / len(val_loader.dataset)
      epoch_mean_corr = running_corr / len(val_loader.dataset)
      
      if epoch_mean_corr > best_corr_val:
          best_corr_val = epoch_mean_corr
          best_corr_model_wts = copy.deepcopy(model.state_dict())
          print('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}, Best Valid Correlation: {:.4f}'
                .format(epoch + 1, best_loss_val, best_corr_val))
    
      if val_loss < best_loss_val:
        best_loss_val = val_loss
        best_loss_model_wts = copy.deepcopy(model.state_dict())
        print('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}, Best Valid Correlation: {:.4f}'
              .format(epoch + 1, best_loss_val, best_corr_val))   
      
      if early_loss_stopping(val_loss) and early_corr_stopping(epoch_mean_corr):
        break

  model.load_state_dict(best_corr_model_wts)
  return model, best_loss_model_wts, best_loss_val

def getBestModelResults(model:nn.Module, val_loader, output_dir, seq_len=1000, n_celltypes=90, bias_model=True):
  """ returns a numpy array of all the predictions"""

  DEVICE = get_least_utilized_gpu() if torch.cuda.is_available() else "cpu"
  print(DEVICE)  # this should print out CUDA 
  model.to(DEVICE)
  scalar_corr, profile_corr, jsd, OCR_profile_corr, OCR_jsd, top_ocr_profile_corr, top_ocr_jsd = model_analysis(
    model, train_loader=val_loader, val_loader=val_loader, test_loader=None, n_celltypes=n_celltypes, 
                 get_scalar_corr=False, get_profile_corr=True, get_jsd=True,
                 eval_set="validation", bin_size=1)
  np.savez(output_dir + '/evaluation', scalar_corr=scalar_corr, profile_corr=profile_corr, 
           jsd=jsd, ocr_profile_corr=OCR_profile_corr, ocr_jsd=OCR_jsd, 
           top_ocr_profile_corr=top_ocr_profile_corr, top_ocr_jsd=top_ocr_jsd)
  return scalar_corr, profile_corr, jsd, OCR_profile_corr, OCR_jsd, top_ocr_profile_corr, top_ocr_jsd 

def select_best_gpu():
  if not torch.cuda.is_available():
    print("No GPU detected. Using CPU instead.")
    return torch.device("cpu")

  device_count = torch.cuda.device_count()
  max_memory = 0
  best_device = 0

  for device_id in range(device_count):
    print("checking device", device_id)
    gpu_memory = torch.cuda.get_device_properties(device_id).total_memory
    if gpu_memory > max_memory:
      max_memory = gpu_memory
      best_device = device_id

  print(f"Selected GPU: {best_device} with {max_memory / (1024**3):.2f} GB memory")
  return torch.device(f"cuda:{best_device}")

def get_least_utilized_gpu():
    if not torch.cuda.is_available():
      print("GPU is not available, using cpu ")
      return torch.device('cpu')

    # Execute the nvidia-smi command and get its output
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Split the output by line and convert to integers (memory usage in MiB)
    gpu_memory_used = list(map(int, output.strip().split('\n')))
    
    # Get the index of the GPU with the least memory usage
    least_utilized_gpu = gpu_memory_used.index(min(gpu_memory_used))

    print(f"Selected GPU: {least_utilized_gpu}")
    return torch.device(f"cuda:{least_utilized_gpu}")

def get_learning_rate_scheduler(scheduler_name: LRSchedulerType, optimizer, **kwargs):
  if scheduler_name == "StepLR":
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
  elif scheduler_name == "CosineAnnealingLR":
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
  elif scheduler_name == "WarmupCosineDecayLR": 
    # The warmup is set to always start at a very small number (lr multiplied by 1e-6) and then increase to whatever the initialized LR was 
    # if total_iters is not given it defaults to 0
    warmup_steps = kwargs.pop('warmup_steps', 0)
    print(warmup_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
      optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    return torch.optim.lr_scheduler.ChainedScheduler(
      schedulers=[warmup_scheduler, cosine_decay_scheduler],
    )
  elif scheduler_name == "Warmup":
    # Only do the warmup without any scheduling after that
    # The warmup is set to always start at a very small number (lr multiplied by 1e-6) and then increase to whatever the initialized LR was 
    # if total_iters is not given it defaults to 0
    warmup_steps = kwargs.pop('warmup_steps', 0)
    return torch.optim.lr_scheduler.LinearLR(
      optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
  else:
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def set_seed():
  seed = 11
  # random.seed(seed)
  # np.random.seed(seed)
  torch.manual_seed(seed)
  print('the seed has been fixed')
    
def main():
  """
  This function is called to train the bpAITAC model

  Arguments:
  1) info_file: path to the info.txt file in the folder produced by prep_data 
              that contains info on the memmaped files of all data 
  2) celltypes - a numpy file with the celltypes in it
  3) length of the sequences in base pairs (ex 2000)
  4) name of the version of the model that is being tested
  5) name of the torch.nn model class that you would like to train (Options are: BPnetRep, and CNN_0)
  6) name of the output directory 
  7) loss function name (options: PoissonNLLLoss, MSELoss, CompositeLoss, CompositeLossBalanced)
  8) number of epochs
  9) _lambda weight on profile head
  10) boolean for bias. If true bias will be included
  11) n_filters number of filters in the body of BPnetRep if this body is being used
  12) ocr_eval: boolean indicating if only the ocr region: middle 250 bp
      should be evaluated when evaluating profile prediction in tran and validation
  13) bin size
  14) bin pooling type either "maxpool" or "avgpool"
  15) batch size
  16) learning rate
  18) name of the learning rate scheduler
  19) learning rate scheduler dictionary of parameters
  20) boolean indicating if the best model should be evaluated (example plots made etc)
  --save_best_loss_model: Optional flag to save the model with the lowest loss. Otherwise defaults to saving model with best corr
  """
  
  parser = argparse.ArgumentParser(description='Train BPnetRep model')
  parser.add_argument('--info_file', help='Path to the info.txt file', required=True)
  parser.add_argument('--celltypes_path', help='Path to celltypes numpy file', required=True)
  parser.add_argument('--seq_len', type=int, help='Length of sequences in base pairs', required=True)
  parser.add_argument('--name', help='Version name of the model being tested', required=True)
  parser.add_argument('--model_name', help='Name of the torch.nn model class', default='BPcm_250')
  parser.add_argument('--output_path', help='Output directory path', required=True)
  parser.add_argument('--loss_fxn', help='Loss function name', default='PoissonLoss')
  parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=200)
  parser.add_argument('--_lambda', type=float, help='Lambda weight on profile head. Only matters when using a composite loss', default=0.5)
  parser.add_argument('--bias', help='Boolean for including bias', default='True')
  parser.add_argument('--n_filters', type=int, help='Number of filters in model body', default=300)
  parser.add_argument('--ocr_eval', help='Boolean for OCR region evaluation', default='False')
  parser.add_argument('--bin_size', type=int, help='Bin size', default=1)
  parser.add_argument('--bin_pooling_type', help='Bin pooling type', default='none')
  parser.add_argument('--scalar_head_fc_layers', type=int, help='Number of scalar head FC layers', default=1)
  parser.add_argument('--batch_size', type=int, help='Batch size', default=20)
  parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
  parser.add_argument('--lr_scheduler_name', help='Learning rate scheduler name', default='Warmup')
  parser.add_argument('--lr_scheduler_args', help='Learning rate scheduler arguments as JSON', default='{"warmup_steps": 1000}')
  parser.add_argument('--set_fixed_seed', help='Boolean for setting fixed seed', default='False')
  parser.add_argument('--memmaped_data', help='Boolean for using memmaped data', default='True')
  parser.add_argument('--save_best_loss_model', action='store_true', default=False, help='DO not save model with best correlation score. instead save model with best loss (default: False)')
  parser.add_argument('--off_by_two', type=str, default='True', help='use this if you are using bp count data that has had an off-by-two correction, and intended output length of your model is smaller than the length of the bp sequences, but the bias data still represents the center of the original sequence that has not been corrected for off-by-two')
  parser.add_argument('--patience', type=int, default=10, help='number of epochs to run without improvment before early stopping')
  args = parser.parse_args()
  model_name = args.model_name
  seq_len=args.seq_len
  n_filters=args.n_filters
  bin_size=args.bin_size
  num_epochs=args.num_epochs
  loss_fxn=args.loss_fxn
  name=args.name
  scalar_head_fc_layers=args.scalar_head_fc_layers
  args.off_by_two = args.off_by_two.lower() == 'true'



  
  print('all args to train.py', sys.argv)
  print("IN TRAIN.py SCRIPT")

  if args.set_fixed_seed.lower() == 'true':
    set_seed()

  # create output directory
  timestamp = datetime.datetime.now().strftime("%m-%d-%Y.%H.%M")
  print("output path: " + args.output_path)

  # Loop until we find a unique timestamp
  while True:
    if os.path.exists(args.output_path + "/" + timestamp):
      time.sleep(60)
      timestamp = datetime.datetime.now().strftime("%m-%d-%Y.%H.%M")
    else:
      print("making a new directory:", args.output_path + "/" + timestamp)
      os.mkdir(args.output_path + "/" + timestamp)
      break

  output_dir = args.output_path + "/" + timestamp
  
  # Convert bin_pooling type to an nn.Module
  if args.bin_pooling_type == 'maxpool':
    bin_pooling_type = nn.MaxPool1d
  elif args.bin_pooling_type == 'avgpool':
    bin_pooling_type = nn.AvgPool1d
  elif args.bin_pooling_type == 'none':
    bin_pooling_type = None
  else:
    print("unsupported bin pooling type")
    return False

  if args.memmaped_data.lower() == 'true':
    print("USING MEMMAPED DATA")
    train_loader, val_loader = load_data(args.info_file, batch_size=args.batch_size)
  else:
    print("In load data non memmaped")
    train_loader, val_loader, test_loader = load_data_non_memmaped(data_config_file=args.info_file, batch_size=args.batch_size)

  celltypes = np.load(args.celltypes_path)
  n_celltypes = celltypes.size
  print("number of celltypes is", celltypes.size)
  if n_celltypes == 1:
    args.save_best_loss_model = True # we want to save the best loss model, because correlation across celltypes will always be zero
    print('args save best loss model', args.save_best_loss_model)

  if model_name =='BPnetRep':
    model = BPnetRep(seq_len, n_celltypes, num_filters=n_filters)
  elif model_name == 'BPmultimax':
    model = BPmultimax(seq_len=seq_len, n_celltypes=n_celltypes)
  elif model_name == 'BPcm':
    model = BPcm(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPcm_250':
    model = BPcm_250(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers, off_by_two=args.off_by_two)
  elif model_name == 'bpAITAC':
    model = bpAITAC(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers, off_by_two=args.off_by_two)
  elif model_name == 'BPcm_bias0':
    model = BPcm_bias0(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPcm_biasRandom':
    model = BPcm_biasRandom(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPcm_noOffTwo':
    model = BPcm_noOffTwo(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPoh':
    model = BPoh(num_filters=n_filters, n_celltypes=n_celltypes, ocr_start=375, ocr_end=625)
  elif model_name == 'BPol':
    model = BPol(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type)
  elif model_name == 'BPmp':
    model = BPmp(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size)
  elif model_name == 'BPcm_skinny':
    model = BPcm_skinny(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPcm_super_skinny':
    model = BPcm_super_skinny(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPbi':
    model = BPbi(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  elif model_name == 'BPbi_shallow':
    model = BPbi_shallow(seq_len=seq_len, num_filters=n_filters, n_celltypes=n_celltypes, bin_size=bin_size, bin_pooling_type=bin_pooling_type, scalar_head_fc_layers=scalar_head_fc_layers)
  else:
    print("model name is incorrect")
    return -1 

  for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

  best_model, best_loss_model_wts, best_loss_val = train(model, train_loader, val_loader, args.loss_fxn, output_dir, 
                                                      epochs=args.num_epochs, _lambda=args._lambda, 
                                                      bias_model=args.bias.lower() == 'true', 
                                                      ocr_eval=args.ocr_eval.lower() == 'true',
                                                      bin_size=args.bin_size, seq_len=args.seq_len,
                                                      learning_rate=args.learning_rate, 
                                                      scheduler_name=args.lr_scheduler_name,
                                                      scheduler_kwargs=json.loads(args.lr_scheduler_args) if args.lr_scheduler_name != "None" else None,
                                                      save_best_loss_model=args.save_best_loss_model,
                                                      off_by_two=args.off_by_two,
                                                      patience=args.patience)

  print("MODEL WAS TRAINED")
  torch.save(best_model.state_dict(), output_dir + '/best_model')
  print("Best correlation model was saved")

  
  model.load_state_dict(best_loss_model_wts)
  torch.save(model.state_dict(), output_dir + '/best_loss_model.pth')

  # make graphs of loss, JSD, and pearson correlation
  plot_training_val_loss(output_dir + '/train_error.txt', output_dir + '/val_error.txt', output_dir, num_epochs, loss_fxn, name, " Training and Validation Loss")
  plot_training_val_loss(output_dir + '/train_jsd.txt', output_dir + '/val_jsd.txt', output_dir, num_epochs, "JSD", name, " Training and Validation JSD")
  plot_training_val_loss(output_dir + '/train_correlation.txt', output_dir + '/val_correlation.txt', output_dir, num_epochs, "Pearson Correlation of Total Counts", name, " Training and Validation Correlation")
  plot_training_val_loss(output_dir + '/train_bp_correlation.txt', output_dir + '/val_bp_correlation.txt', output_dir, num_epochs, "Pearson Correlation of Base-Resolution Counts", name, " Training and Validation Base-Pair Correlation")
  plot_training_val_loss(output_dir + '/train_ocr_jsd.txt', output_dir + '/val_ocr_jsd.txt', output_dir, num_epochs, "OCR JSD", name, " Training and Validation OCR JSD")
  plot_training_val_loss(output_dir + '/train_ocr_bp_correlation.txt', output_dir + '/val_ocr_bp_correlation.txt', output_dir, num_epochs, "Pearson Correlation of OCR Base-Resolution Counts", name, " Training and Validation OCR Base-Pair Correlation")
  plot_training_val_loss(output_dir + '/train_top_ocr_jsd.txt', output_dir + '/val_top_ocr_jsd.txt', output_dir, num_epochs, "JSD of top OCR", name, "Training and Vlidation Top OCR JSD")


  if loss_fxn == "CompositeLoss" or "CompositeLossBalanced" or 'CompositeLossBalancedJSD' or "CompositeLossMNLL":
      plot_training_val_loss(output_dir + '/train_profile_error.txt', output_dir + '/val_profile_error.txt', output_dir, num_epochs, model_name+" profile loss", name, " Profile Loss")
      plot_training_val_loss(output_dir + '/train_scalar_error.txt', output_dir + '/val_scalar_error.txt', output_dir, num_epochs, "MSE", name, " Scalar Loss (MSE)")
  
  print("trained in bin-at-end branch")
  print("made " + output_dir + '/train_bp_correlation.txt')
  # we should aslo save the actual predictions somewhere of the best model 

if __name__=="__main__":
  main()



