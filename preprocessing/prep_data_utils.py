import numpy as np
from collections import defaultdict
import pandas as pd


def quantile_norm(Y: np.ndarray):
    """
    takes the quantile norm of data
    designed to be used for the Tn5 cut data 

    Y: numpy array with dimensions (number of peaks, number of cell types, sequence length)
    """
    # uses scaletype to save memory, without scaletype it would set dtype = float64
    # Y is dtype int16
    scaletype = np.float16
    
    # quantile normalizes counts across output tracks
    Cy = np.sum(Y, axis = -1)
    M = np.mean(np.sort(Cy, axis = 0), axis = 1)
    Cm = M[np.argsort(np.argsort(Cy, axis = 0), axis = 0)]
    Cy[Cy==0] = 1
    C = Cm/Cy
    Y = Y.astype(scaletype)*C[...,None].astype(scaletype)
    return Y

def quantile_norm_center(Y: np.ndarray, ocr_start:int=375, ocr_end:int=625):
    """
    takes the quantile norm of the data across dim = 0 (the OCRs), where the quantile used is just from the center 
    of the sequence. Ex the middle 250 base pairs where the OCR is supposed to be 

    Y: numpy array with dimensions (number of OCRs, number of cell types, sequence length)
    ncenter: the size of the middle section that you would like to take the norm with respect to
    
    returns quantile normalized Y
    and quantile normalized sum of center (total counts)
    """
    # uses scaletype to save memory, without scaletype it would set dtype = float64
    # Y is dtype int16
    scaletype = np.float32

    # quantile normalizes counts across output tracks (cell types)
    Cy = np.sum(Y[...,ocr_start:ocr_end], axis = -1)# sum over middle bp of sequence length 
    M = np.mean(np.sort(Cy, axis = 0), axis = 1) # mean across the diff cell types
    # print('M shape', M.shape)
    Cm = M[np.argsort(np.argsort(Cy, axis = 0), axis = 0)] # revert back to original unsorted order

    Cy[Cy==0] = 1
    C = Cm/Cy # C is the distribution 
    Y = Y.astype(scaletype)*C[...,None].astype(scaletype) # multiply Y by the distribution C to get the quantile normalized version
    
    return Y, Cm


def get_total_ocr_counts(bp_data:np.ndarray, ocr_start=375, ocr_end=625):
    """
    returns the total counts in the OCR region. Note: if seq_len XOR ocr_len is odd, then the total counts returned 
    will be from the OCR region + one additional base pair's read. 
    bp_data: base pair AITAC data, must be of shape: (ocr regions, celltypes, sequence lenghth
    """
    ocr = bp_data[:, :, ocr_start : ocr_end]
    return get_total_counts(ocr)

def get_total_counts(bp_data:np.ndarray):
    return np.sum(bp_data, -1) # sums over the sequence length



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
    print('lineage names', lineage_names)
    print('lineage_cell_indices', lineage_cell_indices)

    return lineage_names, lineage_cell_indices


def average_multi_track(lineage_cell_indices, scalar, profile):
    """
    Take in a specific selected_lineage_name
    """

    num_track_groups = len(lineage_cell_indices)
    # track_group_idx = np.where(lineage_names == selected_lineage_name)
    # track_group = lineage_cell_indices[track_group_idx]
    n_peaks = len(scalar)
    seq_len = profile.shape[-1]
    print('n peaks', n_peaks, 'seq_len', seq_len)
    avg_profile = np.zeros(shape=(n_peaks, num_track_groups, seq_len))
    avg_scalar = np.zeros(shape=(n_peaks, num_track_groups))

    # avg_scalar = np.mean(scalar[:, track_group], dim=1, keepdims=True) # using track_group: track_group + 1 to preserve dim 
    # avg_profile= np.mean(profile[:, track_group, :], dim=1, keepdims=True)
    
    for i, track_group in enumerate(lineage_cell_indices):
        avg_scalar[:, i] = np.mean(scalar[:, track_group], axis=1, keepdims=False)
        avg_profile[:, i, :] = np.mean(profile[:, track_group, :], axis=1, keepdims=False)
    
    return avg_profile, avg_scalar

def average_single_track(selected_lineage_name, lineage_names, lineage_cell_indices, scalar, profile):
    """
    Take in a specific selected_lineage_name
    """
    # num_track_groups = len(lineage_cell_indices)
    print(lineage_cell_indices)
    track_group_idx = np.where([name.lower() == selected_lineage_name.lower() for name in lineage_names])[0][0]

    print('track group idx is', track_group_idx)
    track_group = lineage_cell_indices[track_group_idx]
    print('track_group is', track_group)
    
    # avg_profile = np.zeros(batch_size, num_track_groups, seq_length, device=profile.device)
    # avg_scalar = torch.zeros(batch_size, num_track_groups, device=scalar.device)

    avg_scalar = np.mean(scalar[:, track_group], axis=1, keepdims=True) # using track_group: track_group + 1 to preserve dim 
    avg_profile= np.mean(profile[:, track_group, :], axis=1, keepdims=True)
    
    # for i, track_group in enumerate(self.tracks):
    #     track_group = torch.tensor(track_group, device=profile.device)
    #     avg_scalar[:, i] = torch.mean(scalar[:, track_group], dim=1)
    #     avg_profile[:, i, :] = torch.mean(profile[:, track_group, :], dim=1)
    
    return avg_profile, avg_scalar
