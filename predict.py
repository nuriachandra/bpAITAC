import os
from plotting.combine_results import find_files
from eval_model import get_model_structure
import numpy as np
from utils.inference_utils import load_names, load_observed
from plot_utils_bpaitac import histogram
from utils.region_identification_utils import get_trial_metrics, identify_regions, summary_stats
import pandas as pd
from utils.load_model import get_model, get_predictions, load_model
import scipy
import matplotlib.pyplot as plt
import torch

# Define models

model1_dir = '/data/nchand/analysis/BPcm/BP68_L0_0/'
path_model1 = '/data/nchand/analysis/BPcm/BP68_L0_0/complete/06-12-2024.16.34/best_model'
path_model2 = '/data/nchand/analysis/BPcm/BP68_L-1_5/complete/06-10-2024.00.46/best_model'
model2_dir = '/data/nchand/analysis/BPcm/BP68_L-1_5/'
output_dir = '/data/nchand/analysis/BPcm/BP68_analysis'
model_type = 'BPcm'

n_celltypes = 90
n_filters = 300
seq_len = 998
model_structure = get_model_structure(model_type, n_filters, n_celltypes, seq_len=seq_len)
print(path_model1 == path_model2)
model1 = load_model(path_model1, model_structure=model_structure, n_filters=n_filters, verbose=True)
model2 = load_model(path_model2, model_structure=model_structure, n_filters=n_filters, verbose=False)


# Print model paths again to verify they're different
print(f"Model 1 path: {path_model1}")
print(f"Model 2 path: {path_model2}")

# Verify the perturbation
param1 = next(model1.parameters())
param2 = next(model2.parameters())
print("Before perturbation:", torch.allclose(param1, param2))

# Apply larger perturbation
for param in model2.parameters():
    param.data += torch.randn_like(param) * 0.01

# Check again
param1 = next(model1.parameters())
param2 = next(model2.parameters())
print("After perturbation:", torch.allclose(param1, param2))
# Force models to eval mode
model1.eval()
model2.eval()



### Load data

cell_names = np.load("/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy")

b6_seq_files = np.load('/data/mostafavilab/ImgenATAC/F1_chr11_chr16_seq/b6seqs_padded_chr11chr16.npz')
print(b6_seq_files['names'].shape)
print(b6_seq_files['seqs'].shape)
b6_names = b6_seq_files['names']
b6_seqs = b6_seq_files['seqs']

castseqs_files = np.load('/data/mostafavilab/ImgenATAC/F1_chr11_chr16_seq/castseqs_padded_chr11chr16.npz')
castseqs_names = castseqs_files['names']
castseqs_seqs = castseqs_files['seqs']

### Make predictions


DEVICE = 'cuda'
from tqdm import tqdm
import torch

def pred(model, onehot_seq, n_celltypes=90, batch_size=100):
    torch.cuda.empty_cache()
    model.to(DEVICE)
    counts = torch.zeros((0, n_celltypes)).to(DEVICE)
    onehot_seq = onehot_seq.astype(np.float32)
    seq = torch.from_numpy(onehot_seq)

    # Calculate number of batches (ceiling division)
    n_batches = (len(seq) + batch_size - 1) // batch_size
    
    # Process all batches including the final partial batch
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(seq))
        X = seq[start_idx:end_idx]
        bias = torch.zeros((X.shape[0], X.shape[2]))
        with torch.no_grad():
            profile, scalar = model(X.to(DEVICE), bias.to(DEVICE))
            counts = torch.cat((counts, scalar), 0)
    
    return counts


b6_counts_model1 = pred(model1, b6_seqs)
print(b6_seqs.shape)
print(b6_counts_model1.shape)

b6_counts_model2 = pred(model2, b6_seqs)
print(b6_seqs.shape)
print(b6_counts_model2.shape)


print('out arrays are the same', np.array_equal(b6_counts_model1.cpu(), b6_counts_model2.cpu()))