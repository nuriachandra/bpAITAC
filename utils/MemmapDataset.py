import torch
from torch.utils.data import Dataset
import glob
import numpy as np

class MemmapDataset(Dataset):
    """
    This class represents a custom torch dataset that can be given to a
    dataloader. This dataset works with BOTH memmaped and non-memmaped numpy data

    """

    def __init__(self, genome_one_hot, atac_bp_counts, atac_total_counts, loglikelihood_bias):
        """
        genome_one_hot a np.memmap to the one-hot encoded genomic sequences
        atac_bp_counts: np.memmap to base-pair level counts for each base in the sequence, for all cell types
            this is the data "labels" that the models will train to fit
        atac_total_counts: the sum of all the Tn5 cut counts in each sequence
        loglikelihood_bias: np.memmap to the predicted tn5 cut bias for each sequence
        """

        # initialize variables to keep track of stuff 
        self.one_hot = genome_one_hot
        self.bp_counts = atac_bp_counts
        self.total_counts = atac_total_counts
        self.sequence_bias = loglikelihood_bias
    
    def __getitem__(self, item): 
        # # We transpose the one-hot encoded data to make it (4, seq_len), (4 from 4 nucleotides)
        # # because then it will work with 1d convolutions better
        sequence = torch.from_numpy(self.one_hot[item]).transpose(0,1) 
        profile = torch.from_numpy(self.bp_counts[item])
        total_counts = torch.from_numpy(self.total_counts[item])
        bias = torch.from_numpy(self.sequence_bias[item])
        return sequence.float(), profile.float(), total_counts.float(), bias.float()

    def __len__(self):
        """ Returns the number of peaks represented by this data """
        return self.bp_counts.shape[0] 




