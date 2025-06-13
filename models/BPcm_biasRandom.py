import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
from models.modules import Body, ProfileHead, ScalarHeadMultiMaxpool, ScalarHeadConvMaxpool
from typing import List # use to specify list type

class BPcm_biasRandom(nn.Module):
    """
    This model uses BPnetRep body with 300 filters (as the default)
    and the the ScalarHeadMultiMaxpool head for scalar head
    and the ProfileHead (same as BPnet) for profile head

    bin_size the size that the profile head will be binned into if desired
    scalar_head_fc_layers is for the ScalarHeadConvMaxpool. Traditionally was 1
    """
    def __init__(self, seq_len:int, num_filters:int=300, n_celltypes:int=90, bin_size:int = 1, bin_pooling_type:nn.Module=nn.MaxPool1d, scalar_head_fc_layers:int=1):
        super(BPcm_biasRandom, self).__init__()

        self.num_filters = num_filters
        n_reps = 3 # 3 max pool, conv in scalar head
        pool_size = [5, 5, 5]
        if bin_size == 1:
            pool_size = [5, 5, 5]
        elif bin_size == 2:
            pool_size = [4, 4, 4]
        elif bin_size == 3:
            pool_size = [3, 4, 4]
        elif bin_size == 5:
            pool_size = [3, 3, 3]
        elif bin_size == 10:
            pool_size = [2, 2, 3]
        elif bin_size == 20:
            pool_size = [1, 2, 3]
        else: 
            print('ERROR pool size not implemented for this bin size')
            return 0 

        conv_width = 3
        scalar_head_hidden_layer = 600
        self.body = Body(n_filters=self.num_filters) # can also specify other parameters here
        self.head = BPcmHead(input_channels=num_filters,
                                           seq_length=seq_len, 
                                           n_celltypes=n_celltypes, 
                                           pool_size=pool_size,
                                           n_reps=n_reps,
                                           conv_width=conv_width,
                                           bin_size=bin_size,
                                           scalar_head_fc_layers=scalar_head_fc_layers,
                                           scalar_head_hidden_layer=scalar_head_hidden_layer)
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


class BPcmHead(nn.Module):
    """
    The final layer of the BPmultimax model
    Combines the BPnet profile head and the 
    ScalarHeadMultiMaxpool

    In the final layer, each input track has two outputs:
    1) a prediction of the bp-resolution footprint profile
    2) a prediction of the total count of reads in the window  
    """
    #NOTE see bpnet.heads for reference

    def __init__(self, seq_length:int, n_celltypes:int, input_channels:int, pool_size:List[int], n_reps:int, conv_width:int, bin_size:int, scalar_head_fc_layers:int, scalar_head_hidden_layer:int):
        """
        in_height, in_width: the dimensions of the input feature map
        seq_length: the length in base-pairs of the profile that you want to predict
        celltypes: the different 'tasks' aka celltypes that this model learns to predict for
        input_channels: a.k.a. number of input filters. this is the number of filters used by the 
            preceeding layers 
        conv_width: the width of convolutions used throughout
        """
        super(BPcmHead, self).__init__()

        in_channels = input_channels # this is the number of filters used by the preceeding layers

        self.profile_prediction = ProfileHead(in_channels, n_celltypes, bin_size=bin_size)

        self.total_count_prediction = ScalarHeadConvMaxpool(n_reps=n_reps, 
                                                            in_channels=in_channels,
                                                            conv_width=conv_width,
                                                            seq_len=seq_length, 
                                                            num_classes=n_celltypes,
                                                            pool_size=pool_size,
                                                            n_fc_layers=scalar_head_fc_layers,
                                                            hidden_layer_size=scalar_head_hidden_layer)

    def forward(self, x, bias):
        """
        Returns an two predictions: 
            1) a prediction of the bp-resolution footprint profile
            2) a prediction of the total count of reads in the window 
        x: input feature map to this layer
        """
        print(bias)
        random_bias = torch.randn_like(bias)

        print('RANDOM Balues', random_bias)
        profile = self.profile_prediction(x, random_bias)
        if torch.isnan(torch.mean(profile)):
            print("IN BPcm Head THE PROFILE HAS AN NAN")
            print(profile)
            print(profile.size())
        
        scalar = self.total_count_prediction(x, random_bias)
        return profile, scalar
    