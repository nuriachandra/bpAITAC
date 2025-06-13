import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
from modules import Body, ProfileHead, ScalarHeadConvMaxpool, PoolingBody, ProfileHeadBin
from typing import List # use to specify list type


class BPbin(nn.Module):
    """
    Like BPcm - a BPnetRep model with ScalarHeadConvMaxpool
    but has ability to "bin" the profile data
    """
    
    def __init__(self, seq_len:int, num_filters:int=300, n_celltypes:int=90, bin_size:int=10, pooling_type:nn.Module=nn.MaxPool2D):
        super(BPbin, self).__init__()

        self.num_filters = num_filters
        n_reps = 3 # 3 max pool, conv
        pool_size = 5 # pooling for maxpool conv head
        conv_width = 3
        self.body = PoolingBody(bin_size=bin_size, pooling=pooling_type, n_filters=self.num_filters) # can also specify other parameters here
        self.head = BPbinHead(input_channels=num_filters,
                                           seq_length=seq_len, 
                                           n_celltypes=n_celltypes, 
                                           pool_size=pool_size, # this is for the maxpool conv head
                                           n_reps=n_reps,
                                           conv_width=conv_width,
                                           bin_size=bin_size)
    def forward(self, input, bias):
        """
        returns a tuple containing the output from the head module: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        """
        # check if data is nan
        if (torch.isnan(torch.mean(input))):
            print('input to BPbin is nan')
        x = self.body(input)
        if (torch.isnan(torch.mean(x))):
            print('After Body in BPbin is nan')
        footprint, scalar = self.head(x, bias)
        return footprint, scalar


class BPbinHead(nn.Module):
    """
    The final layer of the BPmultimax model
    Combines the BPnet profile head and the 
    ScalarHeadMultiMaxpool

    In the final layer, each input track has two outputs:
    1) a prediction of the bp-resolution footprint profile
    2) a prediction of the total count of reads in the window  
    """
    #NOTE see bpnet.heads for reference

    def __init__(self, seq_length:int, n_celltypes:int, input_channels:int, pool_size:int, n_reps:int, conv_width:int, bin_size:int):
        """
        input_channels: a.k.a. number of input filters. this is the number of filters used by the 
            preceeding layers 
        conv_width: the width of convolutions used throughout
        """
        super(BPbinHead, self).__init__()

        in_channels = input_channels # this is the number of filters used by the preceeding layers

        self.profile_prediction = ProfileHeadBin(in_channels, n_celltypes, bin_size)
        self.total_count_prediction = ScalarHeadConvMaxpool(n_reps=n_reps, 
                                                            in_channels=in_channels,
                                                            conv_width=conv_width,
                                                            seq_len=seq_length, 
                                                            num_classes=n_celltypes,
                                                            pool_size=pool_size)

    def forward(self, x, bias):
        """
        Returns an two predictions: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        x: input feature map to this layer
        """
        profile = self.profile_prediction(x, bias)
        if torch.isnan(torch.mean(profile)):
            print("IN BPbin Head THE PROFILE HAS AN NAN")
            print(profile)
            print(profile.size())
        scalar = self.total_count_prediction(x, bias)
        return profile, scalar