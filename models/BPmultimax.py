import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
from modules import Body, ProfileHead, ScalarHeadMultiMaxpool
from typing import List # use to specify list type


class BPmultimax(nn.Module):
    """
    This model uses BPnetRep body with 300 filters (as the default)
    and the the ScalarHeadMultiMaxpool
    """
    def __init__(self, seq_len:int, num_filters:int=300, n_celltypes:int=90):
        super(BPmultimax, self).__init__()

        self.num_filters = num_filters
        self.body = Body(n_filters=self.num_filters) # can also specify other parameters here
        pool_size = 5
        n_reps = 3
        self.head = BPmultimaxHead(input_channels=num_filters,
                                           seq_length=seq_len, 
                                           n_celltypes=n_celltypes, 
                                           pool_size=pool_size,
                                           n_reps=n_reps)
    def forward(self, input, bias):
        """
        returns a tuple containing the output from the head module: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        """
        x = self.body(input)
        footprint, scalar = self.head(x, bias)
        return footprint, scalar


class BPmultimaxHead(nn.Module):
    """
    The final layer of the BPmultimax model
    Combines the BPnet profile head and the 
    ScalarHeadMultiMaxpool

    In the final layer, each input track has two outputs:
    1) a prediction of the bp-resolution footprint profile
    2) a prediction of the total count of reads in the window  
    """
    #NOTE see bpnet.heads for reference

    def __init__(self, seq_length:int, n_celltypes:int, input_channels:int, pool_size:int, n_reps:int):
        """
        in_height, in_width: the dimensions of the input feature map
        seq_length: the length in base-pairs of the profile that you want to predict
        celltypes: the different 'tasks' aka celltypes that this model learns to predict for
        input_channels: a.k.a. number of input filters. this is the number of filters used by the 
            preceeding layers 
        """
        super(BPmultimaxHead, self).__init__()

        in_channels = input_channels # this is the number of filters used by the preceeding layers

        self.profile_prediction = ProfileHead(in_channels, seq_length, n_celltypes)
        self.total_count_prediction = ScalarHeadMultiMaxpool(in_channels, 
                                                             seq_len=seq_length, 
                                                             num_classes=n_celltypes,
                                                             pool_size=pool_size,
                                                             n_reps=n_reps)

    def forward(self, x, bias):
        """
        Returns an two predictions: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 

        x: input feature map to this layer
        """
        profile = self.profile_prediction(x, bias)
        scalar = self.total_count_prediction(x, bias)
        return profile, scalar