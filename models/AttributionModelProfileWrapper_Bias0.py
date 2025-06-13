import torch
from torch import nn
from typing import Union
from models.BPcm import BPcm
from BPnetRep import BPnetRep

class AttributionModelProfileWrapper_Bias0(nn.Module):
    def __init__(self, model: Union[BPcm, BPnetRep]):
        super(AttributionModelProfileWrapper_Bias0, self).__init__()
        self.model = model

    def forward(self, input):
        bias = torch.zeros((input.size()[0], input.size()[2]))
        profile, scalar = self.model(input, bias)
        return profile