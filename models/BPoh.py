import torch
from torch import nn
import torch.nn.functional as F # to allow us to define layers
from modules import Body, BasePairHead
from typing import List # use to specify list type


class BPoh(nn.Module):
    """
    This model uses BPnetRep body with 300 filters (as the default)
    and the the ScalarHeadMultiMaxpool
    """
    def __init__(self, num_filters:int=300, n_celltypes:int=90, ocr_start:int=375, ocr_end:int=625):
        super(BPoh, self).__init__()
        self.num_filters = num_filters
        self.body = Body(n_filters=self.num_filters) # can also specify other parameters here
        self.head = BasePairHead(in_channels=num_filters, num_celltypes=n_celltypes, 
                                 ocr_start=ocr_start, ocr_end=ocr_end)

    def forward(self, input, bias):
        """
        returns a tuple containing the output from the head module: 
            1) a prediction of the profile (a probability distribution)
            2) a prediction of the total count of reads in the window 
        """
        x = self.body(input)
        profile, scalar = self.head(x, bias)
        return profile, scalar
