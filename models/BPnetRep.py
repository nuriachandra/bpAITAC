import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
from modules import Body, Head
from typing import List # use to specify list type

class BPnetRep(nn.Module):
    """
    This model replicates the basic structure of BPnet (Avsec et. al 2021)
    using pytorch
    """


    def __init__(self, seq_len:int, n_celltypes:int=90, num_filters=64):
        super(BPnetRep, self).__init__()
        """
        seq_len the number of base-pairs per sequence being used by this model
        celltypes: a list of the names of the different types of cells in the data
        """

        """ The body of BPNet consists of a sequence of convolutional 
        layers with residual skip connections and rectified linear activations. The 
        first convolutional layer uses 64 filters (default) of width 25bp, followed by nine dilated 
        convolutional layers (each with 64 filters (default) of width 3) where the dilation rate 
        (number of skipped positions in the convolutional filter) doubles at every 
        layer. 
        """

        self.num_filters = num_filters
        self.profile_out_len = 250
        self.body = Body(n_filters=self.num_filters) # can also specify other parameters here
        self.head = Head(seq_len, n_celltypes, input_channels=self.num_filters, profile_out_len=self.profile_out_len)
        
     
    def forward(self, input, bias):
        """
        defines the network structure (layer ordering, etc)
        returns a tuple containing the output from the head module: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        """
        x = self.body(input)
        footprint, total = self.head(x, bias)
        return footprint, total