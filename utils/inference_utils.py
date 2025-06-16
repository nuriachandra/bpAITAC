import pandas as pd
import numpy as np
import torch
from utils.MemmapDataset import MemmapDataset
import subprocess
from typing import Union
import os
import yaml
from tqdm import tqdm
from pathlib import Path



DatasetType = Union["train", "test", "validation"]
DataNames = Union["total_counts", "bias", "bp_counts", "onehot"]

def get_least_utilized_gpu():
    assert torch.cuda.is_available(), "GPU is not available, check the directions above (or disable this assertio    n to use CPU)"

    # Execute the nvidia-smi command and get its output
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Split the output by line and convert to integers (memory usage in MiB)
    gpu_memory_used = list(map(int, output.strip().split('\n')))

    # Get the index of the GPU with the least memory usage
    least_utilized_gpu = gpu_memory_used.index(min(gpu_memory_used))

    print(f"Selected GPU: {least_utilized_gpu}")
    return torch.device(f"cuda:{least_utilized_gpu}")


def load_data(info_file:str, batch_size=100):
  """
  Loads data specified in infofile into the dataset,
  and returns DataLoaders containing the datasets for training set and validation set and test set

  info_file: the path to the "info.txt" file in the a folder of data produced by prep_data.py
      info.txt contains file paths and datatypes and shapes for all the data
  batch_size: the batch size to be used in each training iteration.
      defaults to 100
  """
  # load the file as a csv
  info_df = pd.read_csv(info_file, delimiter="\t", names=['path', 'dtype', 'shape'])

  info_df['path'] = info_df['path'].apply(lambda x: x[1:-1]) # git rid of quotation marks in the info_file strings
  info_df['shape'] = info_df['shape'].apply(lambda x: x[1:-1]) # rid of parentheses in 'shapes'

  # 1st column is memmap file path, 2nd column is dtype, 3rd col is shape
  # '/gscratch/mostafavilab/nchand/bpAITAC/data_train_test/complete/memmap/test.names.dat'      dtype('<U26')   (267237,)

  train_df = info_df.iloc[1:5]
  val_df = info_df.iloc[11:15]
  test_df = info_df.iloc[6:10]

  dataframes = [train_df, val_df, test_df]
  train_loader:torch.utils.data.DataLoader = None
  val_loader:torch.utils.data.DataLoader = None
  test_loader:torch.utils.data.DataLoader = None
  dataloader = [train_loader, val_loader, test_loader]

  for j in range(3): 
    df = dataframes[j]
    file_types = [] # [bp_counts, total_counts, bias, onehot]
    # iterate through all data files that need to be included in data loader
    for i in range(4):
      shape = tuple(int(num) for num in df.loc[df.index[i]].at['shape'].split(", ")) # parsing the string representation of the shape
      file_types.append(np.memmap(df.iloc[i,0], dtype='float32', shape=shape)) # loading memory map
    dataset = MemmapDataset(file_types[3], file_types[0], file_types[1], file_types[2]) # [bp_counts, total_counts, bias, onehot]
    dataloader[j] = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)


  return dataloader[0], dataloader[1], dataloader[2] # return train then val data loader then test loader

def load_data_non_memmaped(data_config_file: str, batch_size: int = 100):
    """
    Load numpy arrays from paths specified in a YAML config file for train, validation, and test sets.
    
    Args:
        data_config_file (str): Path to YAML config file containing numpy file paths
        batch_size (int): Batch size for processing (default: 100)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) containing DataLoader objects for each split
    """
    # Load config file
    if not os.path.exists(data_config_file):
        raise FileNotFoundError(f"Config file not found: {data_config_file}")
        
    with open(data_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config is not None (empty YAML file)
    if config is None:
        raise ValueError("Config file is empty")
    
    # Define required keys for each split
    base_keys = ['names', 'onehot', 'total_counts', 'bp_counts', 'bias']
    splits = ['train', 'val', 'test']
    required_keys = []
    for split in splits:
        required_keys.extend([f"{split}_{key}" for key in base_keys])
    
    # Validate required keys exist
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required keys in config file: {missing_keys}")
    
    # Initialize dictionary to store dataloaders
    dataloaders = {}
    
    # Load numpy arrays for each split
    for split in splits:
        try:
            names = np.load(config[f'{split}_names'])
            onehot = np.load(config[f'{split}_onehot'])
            total_counts = np.load(config[f'{split}_total_counts'])
            bp_counts = np.load(config[f'{split}_bp_counts'])
            bias = np.load(config[f'{split}_bias'])
            
            # Validate array shapes are compatible
            if not (len(names) == len(onehot) == len(total_counts) == len(bp_counts) == len(bias)):
                raise ValueError(f"Arrays have inconsistent lengths in {split} split")
            
            # Create dataset object
            dataset = MemmapDataset(
                genome_one_hot=onehot,
                atac_bp_counts=bp_counts,
                atac_total_counts=total_counts,
                loglikelihood_bias=bias
            )
            
            # Create dataloader with appropriate parameters for each split
            shuffle = (split == 'train')  # Only shuffle training data
            dataloaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            
        except Exception as e:
            raise RuntimeError(f"Error loading numpy arrays for {split} split: {str(e)}")
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']


def load_names(info_file, dataset_type: DatasetType):
  # load the file as a csv
  info_df = pd.read_csv(info_file, delimiter="\t", names=['path', 'dtype', 'shape'])

  info_df['path'] = info_df['path'].apply(lambda x: x[1:-1]) # git rid of quotation marks in the info_file strings
  info_df['shape'] = info_df['shape'].apply(lambda x: x[1:-1]) # rid of parentheses in 'shapes'

  # Define a dictionary to map dataset types to the row indices of their name files
  dataset_row_map = {
      "train": 0,
      "test": 5,
      "validation": 10
  }

  # Get the row index based on the dataset_type
  row_index = dataset_row_map[dataset_type]
  # shape = tuple(int(num) for num in info_df.loc[info_df.index[row_index]].at['shape'].split(", ")) # parsing the string representation of the shape
  shape_str = info_df.loc[info_df.index[row_index]].at['shape']
  shape = tuple(int(num.strip()) for num in shape_str.split(",") if num.strip())  # parsing the string representation of the shape
  print('file name', info_df.iloc[row_index,0])
  print('shape', shape)
  names = np.memmap(info_df.iloc[row_index,0], dtype='<U26', shape=shape)
  return names

def load_observed_non_memmaped(info_file, dataset_type: DatasetType, data_name: DataNames):
    """
    Load data from YAML config file using numpy.load (non-memory mapped)
    """
    with open(info_file, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_prefix_map = {
        "train": "train",
        "test": "test", 
        "validation": "val"  # Note: YAML uses 'val' prefix for validation
    }
    data_suffix_map = {
        'total_counts': 'total_counts',
        'onehot': 'onehot', 
        'bias': 'bias',
        'bp_counts': 'bp_counts',
        'peak_names': 'names'  # Note: YAML uses 'names' suffix for peak_names
    }
    
    if dataset_type not in dataset_prefix_map:
        print(f"Unsupported dataset_type: {dataset_type}")
        return None
        
    if data_name not in data_suffix_map:
        print(f"Unsupported data_name: {data_name}")
        return None
    
    # Construct the key for the YAML config
    prefix = dataset_prefix_map[dataset_type]
    suffix = data_suffix_map[data_name]
    key = f"{prefix}_{suffix}"
    
    if key not in config:
        print(f"Key '{key}' not found in config file")
        return None
    
    file_path = config[key]
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return None
    
    print(f'Loading file: {file_path}')
    
    # Load data using numpy.load (non-memory mapped)
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_observed(info_file, dataset_type: DatasetType, data_name: DataNames):
  if info_file.lower().endswith('.yaml') or info_file.lower().endswith('.yml'):
    return load_observed_non_memmaped(info_file, dataset_type, data_name)

  info_df = pd.read_csv(info_file, delimiter="\t", names=['path', 'dtype', 'shape'])
  info_df['path'] = info_df['path'].apply(lambda x: x[1:-1]) # git rid of quotation marks in the info_file strings
  info_df['shape'] = info_df['shape'].apply(lambda x: x[1:-1]) # rid of parentheses in 'shapes'

  # Define a dictionary to map dataset types to the row indices of their name files
  total_counts_row_map = {
      "train": 2,
      "test": 7,
      "validation": 12
  }

  onehot_row_map = {
      "train": 4,
      "test": 9,
      "validation": 14
  }

  bias_row_map = {
      "train": 3,
      "test": 8,
      "validation": 13
  }

  bp_counts_row_map = {
      "train": 1,
      "test": 6,
      "validation": 11
  }

  names_row_map = {
      "train": 0,
      "test": 5,
      "validation": 10
  }
  dtype='float32'
  if data_name == 'total_counts':
    dataset_row_map = total_counts_row_map
  elif data_name == 'onehot':
    dataset_row_map = onehot_row_map
  elif data_name == 'bias':
    dataset_row_map = bias_row_map
  elif data_name == 'bp_counts':
    dataset_row_map = bp_counts_row_map
  elif data_name == 'peak_names':
    dataset_row_map = names_row_map
    dtype='<U26'
  else:
    print(data_name, "Unsupported data_name. This method may not be yet complete")
    return
  
  # Get the row index based on the dataset_type
  row_index = dataset_row_map[dataset_type]
  # shape = tuple(int(num) for num in info_df.loc[info_df.index[row_index]].at['shape'].split(", ")) # parsing the string representation of the shape
  shape_str = info_df.loc[info_df.index[row_index]].at['shape']
  shape = tuple(int(num.strip()) for num in shape_str.split(",") if num.strip())  # parsing the string representation of the shape
  print('file name', info_df.iloc[row_index,0])
  data = np.memmap(info_df.iloc[row_index,0], dtype=dtype, shape=shape)
  return data


def predict_scalar(model, onehot_seq, n_celltypes=90, batch_size=100, device='cuda'):
    """
    onehot_seq has shape (n, 4, seq_len)
    """
    torch.cuda.empty_cache()
    model.to(device)
    counts = torch.zeros((0, n_celltypes)).to(device)
    if isinstance(onehot_seq, np.ndarray):
        onehot_seq = onehot_seq.astype(np.float32)
        seq = torch.from_numpy(onehot_seq)
    else:
       seq = onehot_seq

    # Calculate number of batches (ceiling division)
    n_batches = (len(seq) + batch_size - 1) // batch_size
    
    # Process all batches including the final partial batch
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(seq))
        X = seq[start_idx:end_idx]
        bias = torch.zeros((X.shape[0], X.shape[2]))
        with torch.no_grad():
            profile, scalar = model(X.to(device), bias.to(device))
            counts = torch.cat((counts, scalar), 0)
    
    return counts


def predict_all(model, onehot_seq, out_seq_len=998, n_celltypes=90, batch_size=100, device='cuda'):
    """
    onehot_seq has shape (n, 4, seq_len)
    """
    torch.cuda.empty_cache()
    model.to(device)
    counts = torch.zeros((0, n_celltypes)).to(device)
    profile_pred = torch.zeros((0, n_celltypes, out_seq_len)).to(device)
    if isinstance(onehot_seq, np.ndarray):
        onehot_seq = onehot_seq.astype(np.float32)
        seq = torch.from_numpy(onehot_seq)
    else:
       seq = onehot_seq
       
    # Calculate number of batches (ceiling division)
    n_batches = (len(seq) + batch_size - 1) // batch_size
    
    # Process all batches including the final partial batch
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(seq))
        X = seq[start_idx:end_idx]
        bias = torch.zeros((X.shape[0], X.shape[2]))
        with torch.no_grad():
            profile, scalar = model(X.to(device), bias.to(device))
            counts = torch.cat((counts, scalar), 0)
            profile_pred = torch.cat((profile_pred, profile), 0)
    
    return profile_pred, counts
