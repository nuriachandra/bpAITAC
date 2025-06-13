import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
from models.modules import Body, ProfileHead, ScalarHeadMultiMaxpool, ScalarHeadConvMaxpool, ScalarHeadOneLayer, ScalarHeadMaxPool
from typing import List # use to specify list type

class BPmp(nn.Module):
    """
    This model is the same as BPcm, except for the scalar head, which has a single linear layer
    This model uses BPnetRep body with 300 filters (as the default)
    and the the ScalarHeadOneLayer head for scalar head
    and the ProfileHead (same as BPnet) for profile head

    bin_size the size that the profile head will be binned into if desired
    scalar_head_fc_layers is for the ScalarHeadConvMaxpool. Traditionally was 1
    """
    def __init__(self, seq_len:int, num_filters:int=300, n_celltypes:int=90, bin_size:int = 1):
        super(BPmp, self).__init__()

        self.num_filters = num_filters
        n_reps = 3 # 3 max pool, conv in scalar head
        pool_size = [5, 5, 5]
        conv_width = 3
        self.profile_out_len=250
        self.body = Body(n_filters=self.num_filters) # can also specify other parameters here
        self.head = BPmpHead(input_channels=num_filters,
                                           seq_length=seq_len, 
                                           n_celltypes=n_celltypes, 
                                           pool_size=pool_size,
                                           n_reps=n_reps,
                                           conv_width=conv_width,
                                           bin_size=bin_size,
                                           profile_out_len=self.profile_out_len)
    def forward(self, input, bias):
        """
        returns a tuple containing the output from the head module: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        """
        # check if data is nan
        if (torch.isnan(torch.mean(input))):
            print('input to BPcm is nan')
        x = self.body(input)
        if (torch.isnan(torch.mean(x))):
            print('After Body in BPcm is nan')
        footprint, scalar = self.head(x, bias)
        return footprint, scalar


class BPmpHead(nn.Module):
 
    #NOTE see bpnet.heads for reference

    def __init__(self, seq_length:int, n_celltypes:int, input_channels:int, pool_size:List[int], n_reps:int, conv_width:int, bin_size:int, profile_out_len:int=998):
        """
        in_height, in_width: the dimensions of the input feature map
        seq_length: the length in base-pairs of the profile that you want to predict
        celltypes: the different 'tasks' aka celltypes that this model learns to predict for
        input_channels: a.k.a. number of input filters. this is the number of filters used by the 
            preceeding layers 
        conv_width: the width of convolutions used throughout
        """
        super(BPmpHead, self).__init__()

        in_channels = input_channels # this is the number of filters used by the preceeding layers

        self.profile_prediction = ProfileHead(in_channels, n_celltypes, bin_size=bin_size, profile_out_len= profile_out_len)

        self.total_count_prediction = ScalarHeadMaxPool(pool_size=50,
                                                        in_channels=in_channels,
                                                        seq_len=seq_length, 
                                                        num_classes=n_celltypes)

    def forward(self, x, bias):
        """
        Returns an two predictions: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        x: input feature map to this layer
        """
        profile = self.profile_prediction(x, bias)
        if torch.isnan(torch.mean(profile)):
            print("IN BPcm Head THE PROFILE HAS AN NAN")
            print(profile)
            print(profile.size())
        scalar = self.total_count_prediction(x, bias)
        return profile, scalar
    