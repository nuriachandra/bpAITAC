import torch
from torch import nn

class AttributionModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(AttributionModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        bias = torch.zeros((input.size()[0], input.size()[2]))
        profile, scalar = self.model(input, bias)
        return scalar