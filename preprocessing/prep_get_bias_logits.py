import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
from eval_model import get_model_structure
from  utils.load_model import load_model, DEVICE

def get_model(saved_model_path, model_structure_name='BPbi', seq_len=1000):
    structure = get_model_structure(model_structure_name, 300, 1, seq_len=seq_len)
    model = load_model(saved_model_path=saved_model_path, n_celltypes=1, n_filters=300, model_structure=structure, verbose=True)
    return model

def get_logits(model, onehot_npz_path, out_seq_len=250):
    """
    Return numpy arrays of names and logits for all sequences in the input file
    
    Parameters:
    -----------
    model : torch.nn.Module
        The loaded model to use for predictions
    onehot_npz : str
        Path to npz file containing onehot encoded sequences
        
    Returns:
    --------
    tuple(np.ndarray, np.ndarray)
        Returns (names, logits) where names are the gene names and 
        logits are the model predictions for each sequence
    """
    d = np.load(onehot_npz_path, allow_pickle=True)
    onehots = d['seqfeatures'][0]
    onehots = np.transpose(onehots, (0,2,1))

    names = d['genenames']

    print("Shape:", onehots.shape)
    print("Dtype:", onehots.dtype)
    print("Sample element shape:", onehots[0].shape)  
    
    # Get the shape of a single prediction to initialize the array

    # Initialize array to store all logits
    all_logits = np.zeros((len(onehots), 1, out_seq_len))
    
    # Process each sequence
    for i in tqdm(range(len(onehots)), desc="Processing sequences", unit="sequence"):
        region_onehot = torch.from_numpy(onehots[i:i+1]).float().to(DEVICE)
        batch_logits = model.predict_logits(region_onehot)  # Preserve first dimension
        all_logits[i] = batch_logits[0].cpu().detach().numpy()  # Remove batch dimension when storing

    
    return names, all_logits

def main():
    '''
    Example of how to run:
    python prep_get_bias_logits.py --onehot_npz /data/nchand/mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz --saved_bias_model /homes/gws/nchand/MostafaviLab/results/BPbi/BP106_L0_1/complete/11-02-2024.14.48/best_model --out_path /data/nchand/ImmGen/mouse/BPprofiles1000/bias/BP106_L0_1_deprotinated_mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz
    '''
    parser = argparse.ArgumentParser(description='Process onehot npz and saved bias model')
    
    parser.add_argument(
        '--onehot_npz',
        type=str,
        required=True,
        help='Path to the one-hot encoded npz file'
    )
    
    parser.add_argument(
        '--saved_bias_model',
        type=str,
        required=True,
        help='Path to the saved bias model'
    )

    parser.add_argument('--out_path', type=str, required=True, help='Where the resultant npz file will be saved')
    
    args = parser.parse_args()
    saved_bias_model_path = args.saved_bias_model

    model = get_model(saved_model_path=saved_bias_model_path)
    names, bias_logits = get_logits(model, args.onehot_npz)
    
    # Save the arrays to npz file
    np.savez(args.out_path, counts=bias_logits, names=names)

if __name__ == "__main__":
    main()
