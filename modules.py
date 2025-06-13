import torch
from torch import nn
import torch.nn.functional as F 
from typing import List # use to specify list type
from functions import JSD, normalize


class ResBlock(nn.Module):
    """
    This class represents a 1D convolutional block with a rectified 
    linear activation function and with a residual skip. 

    Note: I am currently using the built in ResConv1d in place of this one
    """

    def __init__(self, in_channels:int, filter_width:int, dilation:int, num_filters:int):
        """
        in_channels: integer number of channels in the input
        filter_width: integer number of base-pairs that each filter covers
        dilation: integer dilation rate, a.k.a. number of skipped 
                    positions in each convolutional filter
        """
        super(ResBlock, self).__init__()
        # padding_len: int = dilation * (filter_width - 1) / 2 # NOTE: must change to floor or cieling if dialation is not multiple of two
        self.conv_block = nn.Sequential(
            # input size not expected:
            # Given groups=1, weight of size [64, 1, 25], expected input[100, 1000, 4] to have 1 channels, but got 1000 channels instead
            
            # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv1d(in_channels=in_channels, out_channels=num_filters, 
                        kernel_size=filter_width,  padding='same', dilation=dilation),
            nn.BatchNorm1d(num_filters),
            nn.ReLU())
            
    
    def forward(self, x):
        """ 
        The forward propagation step that can be called 
        to run through this ResBlock. Returns the sum of the output 
        from this 1d convolution and the input
        x: the input to this block 
        """
        # sum input with the predition from nn.Sequential conv_block
        return x + self.conv_block(x)
        


class DialatedConvs(nn.Module):
    """
    This class represents a series of dialated convolutional layers
    with residual skipping in where the dilation rate doubles at every layer.
    There are ReLU functions in between convolutions 
    """
    def __init__(self, in_channels:int, n_layers:int, filter_width=3, n_filters=64, start_dilation=2):
        """
        Initializes the DialatedConvs module
        in_channels: number of input channels # TODO possible change to 64 constant
        n_layers: the number of dialated convolutional layers
        """
        super(DialatedConvs, self).__init__()
        # net is a sequential module holding all layers
        self.filter_width = filter_width
        self.num_filters = n_filters
        self.net = self.initializeLayers(in_channels, n_layers, start_dilation) 

    def initializeLayers(self, in_channels:int, n_layers:int, start_dilation) -> nn.Sequential:
        """
        Initializes the residual skip 1d convolutional layers that form the
            body of the DialatedConvs class. The first layer has dilation rate = 2
            and each following layer has double the dilation rate of the previous layer 
        Returns nn.Sequential module containing all of the layers

        in_channels: number of input channels # TODO possible change to 64 constant
        n_layers: the number of layers to initialize
        """
        dilation: int = start_dilation # dilation starts at 2 and doubles each time
        layers = []
        # add each layer to the list
        for i in range(n_layers):
            layers += [ResBlock(in_channels, self.filter_width, dilation, num_filters=self.num_filters)]
            dilation = dilation*2 
            in_channels = self.num_filters # in layers after the first one, the input channels equal the num filters
        # convert the list of layers to a sequential module of the layers
        return nn.Sequential(*layers)

    def forward(self, x):
        """ x: input to the convolutional layers """
        return self.net(x)
        


class Body(nn.Module):
    """ 
    This class represents the entire body of BPnet, which includes 
    all layers except for the output layer.

    bin_size the number of bps of profile data that will be binned
    """

    def __init__(self, num_dialated_layers:int=9, conv_width:int=25, n_filters:int=64):
        super(Body, self).__init__()
        in_channels:int = 4 # the number of channels in the input data is 4 becuase it is a one-hot sequence of 4 nucleotides"""
        first_conv_width: int = conv_width # the first convolutional layer has a width of 25 bp
        n_dialated_layers: int = num_dialated_layers # the number of dialated conv layers
        
        layers = [nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=first_conv_width, padding='same')]
        layers.append(DialatedConvs(in_channels=n_filters, n_layers=n_dialated_layers, n_filters=n_filters))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """ x: input data """
        return self.net(x)

class ModularBody(nn.Module):
    """
    This class represents a modular convolutional network body with residual connections 
    """

    def __init__(self, num_layers=3, n_filters=300, c1_width=25, cx_width=5, start_dilation=2):
        super(ModularBody, self).__init__()
        in_channels:int = 4 # the number of channels in the input data is 4 becuase it is a one-hot sequence of 4 nucleotides"""
        first_conv_width: int = c1_width # the first convolutional layer has a width of 25 bp
        n_dialated_layers: int = num_layers - 1  # the number of dialated conv layers
   
        self.net = nn.Sequential(
            # ResBlock(in_channels, first_conv_width, dilation=1, num_filters=n_filters), 
            # lets try the first layer as just a conv1d w/o dialation or residual skip
            # TODO optimize kernel size - edit in Dialated convs
            nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=first_conv_width, padding='same'), #  first layer as just a conv1d w/o dialation or residual skip
            DialatedConvs(in_channels=n_filters, n_layers=n_dialated_layers, filter_width=cx_width, n_filters=n_filters, start_dilation=start_dilation)
        )

    def forward(self, x):
        """ x: input data """
        return self.net(x)

class PoolingBody(nn.Module):
    """
    This class represents the BPnetBody with added pooling to the first several layers 

    bin_size: the number of bps that the provile will be binned into 
    pooling: either maxpool2D or AvgPool2D. No other inputs are supported
    """
    def __init__(self, bin_size, pooling:nn.Module, num_dialated_layers:int=9, conv_width:int=25, n_filters:int=300):
        super(PoolingBody, self).__init__()
        in_channels:int = 4
        first_conv_width: int = conv_width 
        n_dialated_layers: int = num_dialated_layers 

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=first_conv_width, padding='same'),
            pooling(kernel_size=bin_size, stride=None, padding=0, ceil_mode=False),
            DialatedConvs(in_channels=n_filters, n_layers=n_dialated_layers, n_filters=n_filters)
        )


    def forward(self, x):
        return self.net(x)

class ProfileHead(nn.Module):
    """ 
    The final layer that outputs a base-pair resolution profile prediction
    
    Uses a deconvolution / transposed convolution to create an output 
    of the same dimension as the original input sequence 
    predictsthe strand-specific probabilities of observing a 
    particular read at a particular position in the input sequence
    uses softmax to output the probability distribution along the sequence
    in the output, each channel represents a different task/cell type

    takes the input data and the bias associated with that sequence 
    and adds the bias before taking the softmax
    """

    def __init__(self, in_channels:int, num_classes:int, bin_size:int, profile_out_len:int=1000, off_by_two:bool=False):
        """ 
        in_channels: the number of channels in the input data 
        seq_length: the length in base-pairs of the profile that you want to predict
        num_classes: the number of celltypes / output classes
        """
        super(ProfileHead, self).__init__()

        # in bpnet the number of filters is the number of tracks per task which is 2
        # tconv_kernel shape = (25, 1) 
        # padding = 'same'
        #  (1) a deconvolutional layer (filter width 25—a typical ChIP–nexus 
        # footprint width) predicting the strand-specific probabilities of observing a 
        # particular read at a particular position in the input sequence (shape or profile 
        # prediction)
        self.off_by_two = off_by_two
        self.bin_size = bin_size
        self.bin = Bin(bin_size, dim=2) # dim 2 has the sequence length prediction
        self.profile_out_len = profile_out_len
        kernel_size = 25 # TODO optimize - bp net used this bc 25bp is typical ChIP nexus width
    
        # we want the output size to be the same as the sequence_length
        # Lout = (Lin−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
        # seq_len = seq_len - 1 - 2xpadding + kernel_size - 1 + output_padding + 1
        # 2 * padding =  - 1  + kernel_size 
        # padding: int = ( kernel_size - 1 ) // 2 # trying truncated division, we shall see if it works if kernel size is even

        self.deconvolution = nn.Conv1d(in_channels, out_channels=num_classes, kernel_size=kernel_size, padding='same') # we need to do this for every cell type 
            # trying if output channels = cell_type.length?
        self.softmax = nn.Softmax(dim=2) # dim 2 has the sequence lengths, and we want to softmax over the sequence
        

    
    def trim_to_center(self, x):
        last_dim_size = x.shape[-1] 
        if self.off_by_two: # correct off-by-two to find the center 250 bp if the full seq len was 1000 instead of 998
            last_dim_size += 2
        start_idx = (last_dim_size - self.profile_out_len) // 2 
        end_idx = start_idx + self.profile_out_len
        out = x[..., start_idx:end_idx]
        return out

    
    def forward(self, x, bias):
        """
        x: the input data to this layer
        bias: the bias associated with this sequence that x is derived from
        """
        # bias is currently a double tensor so we cast to a float - THIS COULD BE A SOURCE OF small ERROR, 
        bias = bias.float() 
        # returns a tensor with probability at each location along sequence 
        out = self.deconvolution(x)
        if torch.isnan(torch.mean(out)):
            print("OUT of Deconvolution has an nan")

        if self.profile_out_len < out.size(-1):
            # Get the center 250 elements from the last dimension
            out = self.trim_to_center(out)
        
        if self.profile_out_len < bias.size(-1):
            bias = self.trim_to_center(bias)

        bias_corrected = out + torch.unsqueeze(bias, dim=1)
        if torch.isnan(torch.mean(bias_corrected)):
            print("bias_corrected has an nan")
        # bin the output
        if self.bin_size != 1:
            bias_corrected = self.bin(bias_corrected)
        profile = self.softmax(bias_corrected)
        if torch.isnan(torch.mean(profile)):
            print("profile out of softmax has an nan")
        return profile
    
    def predict_logits(self, x):
        """
        This method predicts the logits (before softmax and before bias is added)
        """
        out = self.deconvolution(x)
        # get the center 250 bp
        out = self.trim_to_center(out)
        return out

    

class ProfileHeadConvs(nn.Module):
  """ a series of convolutions with residual connections
      ending with a softmax output to predict profile"""

  def __init__(self, n_layers:int, in_channels:int, sequence_len:int, num_classes:int, 
              conv_width:int, num_filters:int, start_dilation:int):
    super(ProfileHeadConvs, self).__init__()
    dilation = start_dilation
    layers = []
    # add each layer to the list
    for i in range(n_layers-1):
      layers += [ResBlock(in_channels, conv_width, dilation, num_filters=num_filters)]
      dilation = dilation*2 
      in_channels = num_filters # in layers after the first one, the input channels equal the num filters
      dilation = dilation * 2
    # convert the list of layers to a sequential module of the layers
    self.net = nn.Sequential(*layers)
    self.deconvolution = nn.Conv1d(in_channels, out_channels=num_classes, kernel_size=conv_width, padding='same')
    self.softmax = nn.Softmax(dim=2) # dim 2 has the sequence lengths, and we want to softmax over the sequence

  def forward(self, x, bias):
    # bias is currently a double tensor so we cast to a float - THIS COULD BE A SOURCE OF small ERROR, 
    bias = bias.float() 
    # returns a tensor with probability at each location along sequence 
    out = self.net(x)
    out = self.deconvolution(x)
    bias_corrected = out + torch.unsqueeze(bias, dim=1)
    return self.softmax(bias_corrected)

        
    

class ScalarHead(nn.Module):
    """ 
    A final layer for bpnet that outputs the predicted average count of fragments in a region
    
    Uses global average pooling to get a predicted count estimation
    Contains the bottlenck in the global average pool
    """

    def __init__(self, in_channels:int, num_classes:int):
        """
        in_channels: the number of input channels comining into the head 
        num_classes: the number of celltypes / output classes
        """
        super(ScalarHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # avg pool across sequence (the last dim becomes 1)
        self.output_layer = nn.Linear(in_features=in_channels, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, bias):
        """
        x is the input feature map to this layer
        Takes the mean average of all elements in each seq_len feature map
        then uses a fully connected layer to get output size for each cell type

        NOTE: this method assumes that the sequence is 1000 bp and has a 250 bp ocr in the center

        output will be a tensor of shape (batch_size, number of cell_types)
        """        
        # averaged = F.avg_pool1d(x, kernel_size=x.size()[-1]) # avg pool across sequence 
        averaged = self.avg_pool(x)  # avg pool across sequence 
        averaged = torch.squeeze(averaged, dim=-1)
        out = self.output_layer(averaged) # apply FC layer and ReLU
        # output = out 
        output = self.relu(out)
        return output

class ScalarHeadMaxPool(nn.Module):
    """ 
    A final layer for bpnet that outputs the predicted average count of fragments in a region
    
    Uses a max pooling and then a fully-connected layer to get a predicted count estimation
    The max pooing has stride length = kernel length, so there is no overlap
    """
    def __init__(self, in_channels:int, num_classes:int, pool_size=100, seq_len=1000):
        """
        in_channels: the number of input channels comining into the head 
        num_classes: the number of celltypes / output classes
        """
        super(ScalarHeadMaxPool, self).__init__()
        self.pool_layer = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        
        num_bins = seq_len // pool_size  # floor fxn is appropriate 
        self.output_layer = nn.Linear(in_features=num_bins*in_channels, out_features=num_classes)
            

    def forward(self, x, bias=None):
        # print(x.size())
        out = self.pool_layer(x)
        # print(out.size())
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        # print(out.size())
        out = self.output_layer(out)
        # print(out.size())
        # print(out)
        out = F.relu(out)
        return out
    
class ScalarHeadOneLayer(nn.Module):
    """ 
    A final layer for bpnet that outputs the predicted average count of fragments in a region
    
    Uses a max pooling and then a fully-connected layer to get a predicted count estimation
    The max pooing has stride length = kernel length, so there is no overlap
    """
    def __init__(self, in_channels:int, num_classes:int, seq_len=1000):
        """
        in_channels: the number of input channels comining into the head 
        num_classes: the number of celltypes / output classes
        """
        super(ScalarHeadOneLayer, self).__init__()
        self.output_layer = nn.Linear(in_features=seq_len*in_channels, out_features=num_classes)
            

    def forward(self, x, bias=None):
        # print(out.size())
        out = torch.flatten(x, start_dim=1, end_dim=-1)
        out = self.output_layer(out)
        # print(out)
        out = F.relu(out)
        return out

class ScalarHeadMultiMaxpool(nn.Module):
    """
    Maxpool, fully connected layer, relu layer repeated
    ending in a fully connected layer outputing a scalar prediction

    in_channels: the number of filters/ channels coming in from the body 
        preceeding this head
    num_classes: the number of classes/celltypes. This module predicts 
        a scalar value for each of the classes
    pool_size: This is the pool size for all pooling layers. 
        The stride size will be set as the same as the pool size. 
        Default 5
    n_reps: the number of times that  Maxpool, fully connected layer, relu 
        is repeated before the final fully connected layer. Default 3 times
    """
    def __init__(self, in_channels:int, seq_len:int, num_classes:int, pool_size:int=5, n_reps:int=3):
        super(ScalarHeadMultiMaxpool, self).__init__()
        # starts by flattening the input
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
 
        layers = [nn.Flatten(start_dim=1, end_dim=-1)]
        pool_size = [200, 5, 5]
        len = in_channels*seq_len
        print("start len", len)
        for i in range(n_reps):
            layers += [nn.MaxPool1d(kernel_size=pool_size[i], stride=pool_size[i])]
            len = (len - pool_size[i]) // pool_size[i] + 1 # (len - kernel size)//stride +1
            print("len at it", i, ":", len)
            layers += [nn.Linear(in_features=len, out_features=len)]
            layers += [nn.ReLU()]
        # add final fully connected layer
        print("len here", len)
        layers += [nn.Linear(in_features=len, out_features=num_classes)]
        # add final ReLU to make sure all predictions are positive
        layers += [nn.ReLU()]
        print("layers", layers)
        self.all_layers = nn.Sequential(*layers)

    
    def forward(self, input, bias=None):
        # x = torch.flatten(input, start_dim=1, end_dim=-1)
        # len = self.in_channels*self.seq_len
        # print("actual shape 1:", x.size(), "predicted len:", len)
        # x = F.max_pool1d(x, kernel_size=5, stride=5)
        # len = (len - 5) / 5 + 1
        # print("actual shape 2:", x.size(), "predicted len:", len)
        # x = self.fc(x)
        out = self.all_layers(input)
        return out

class ScalarHeadConvMaxpool(nn.Module):
    """ 
    maxpool followed by convolutions n_reps times
    pool_size a list on integers as long as n_reps corresponding to the size
        of poolint after each convolution
    """
    def __init__(self, n_reps, in_channels:int, num_classes:int, 
                 pool_size:List[int], conv_width:int, seq_len:int, 
                 n_fc_layers=1, hidden_layer_size=300):
      super(ScalarHeadConvMaxpool, self).__init__()
      
      # convert the list of layers to a sequential module of the layers
      self.net, out_len = self.initialize_layers(n_reps=n_reps, 
                                                 in_channels=in_channels, 
                                                 conv_width=conv_width, 
                                                 pool_size=pool_size, 
                                                 seq_len=seq_len)
      fc_layers = [nn.Flatten(start_dim=1, end_dim=-1)]
      fc_in_features = out_len*in_channels
      for i in range(n_fc_layers-1):
        fc_layers += [nn.Linear(in_features=fc_in_features, out_features=hidden_layer_size)]
        fc_layers += [nn.ReLU()]
        fc_in_features = hidden_layer_size 
        hidden_layer_size = hidden_layer_size // 2 # reduce size of layer by half each time 
      # add final output layer
      fc_layers += [nn.Linear(in_features=fc_in_features, out_features=num_classes)] 
      fc_layers += [nn.ReLU()] # end with a ReLU to make results positive
      self.out_layers = nn.Sequential(*fc_layers)

    def initialize_layers(self, n_reps, in_channels:int, conv_width:int, pool_size:List[int], seq_len:int):
      # add each layer to the list
      layers = []
      for i in range(n_reps):
        layers += [nn.MaxPool1d(kernel_size=pool_size[i], stride=pool_size[i])]
        seq_len = seq_len // pool_size[i]
        print(seq_len)
        layers += [nn.Conv1d(in_channels=in_channels, out_channels=in_channels, 
                    kernel_size=conv_width, padding='same')]
        # add batchnorm
        layers += [nn.BatchNorm1d(in_channels)] # in channels because conv size is constant
        layers += [nn.ReLU()]
      return nn.Sequential(*layers), seq_len

    def forward(self, input, bias=None):
      out = self.net(input)
      if torch.isnan(torch.mean(out)):
        print("there is nan in scalar after the convs")
        exit()
      out = self.out_layers(out)
      if torch.isnan(torch.mean(out)):
        print("there is nan in scalar out")
        exit()
      return out


class ScalarHeadFCLayers(nn.Module):
    """ Three fully connected layers like in AI-TAC for the head """
    def __init__(self, in_channels:int, out_features:int, seq_len=1000, internal_features:int=1000):
        super(ScalarHeadFCLayers, self).__init__()

        self.fc_layer1 = nn.Sequential(
            nn.Linear(in_features=in_channels*seq_len,
                      out_features=internal_features),
            nn.ReLU(),
            nn.Dropout(p=0.03))
        self.fc_layer2 = nn.Sequential(
            nn.Linear(in_features=internal_features,
                      out_features=internal_features),
            nn.ReLU(),
            nn.Dropout(p=0.03))
        self.fc_layer3 = nn.Sequential(
                nn.Linear(in_features=internal_features,
                          out_features=out_features))

    def forward(self, x, bias):
        """ x is output from convolutions """
        input = torch.flatten(x, start_dim=1, end_dim=-1)
        out = self.fc_layer1(input)
        out = self.fc_layer2(out)
        out = self.fc_layer3(out)
        return F.relu(out)

class AITACHead(nn.Module):
    """ MaxPool thaen Three fully connected layers like in AI-TAC for the head """
    def __init__(self, in_channels:int, out_features:int, pool_size=100):
        super(AITACHead, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.fc_layers = ScalarHeadFCLayers(in_channels, out_features, seq_len=10)

    def forward(self, x, bias):
        """ x is output from convolutions """
        out = self.maxpool(x)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.fc_layers(out, bias)
        return F.relu(out)



class Head(nn.Module): 
    """ 
    This class represents the final layer of BPnet. 
    
    In the final layer, each input track has two outputs:
    1) a prediction of the bp-resolution footprint profile
    2) a prediction of the total count of reads in the window  
    """
    #NOTE see bpnet.heads for reference

    def __init__(self, seq_length:int, n_celltypes:int, input_channels:int=64, profile_out_len:int=998):
        """
        in_height, in_width: the dimensions of the input feature map
        seq_length: the length in base-pairs of the profile that you want to predict
        celltypes: the different 'tasks' aka celltypes that this model learns to predict for
        input_channels: a.k.a. number of input filters. this is the number of filters used by the 
            preceeding layers 
        """
        super(Head, self).__init__()

        in_channels = input_channels # this is the number of filters used by the preceeding layers

        self.profile_prediction = ProfileHead(in_channels, n_celltypes, bin_size=1, profile_out_len=profile_out_len)
        # BPnet head:
        self.total_count_prediction = ScalarHead(in_channels, n_celltypes)
        # MaxPool head:
        # self.total_count_prediction = ScalarHeadMaxPool(in_channels, len(celltypes))
        # just FC layers:
        # self.total_count_prediction = ScalarHeadFCLayers(in_channels, len(celltypes))
        # AI-TAC head:
        # self.total_count_prediction = AITACHead(in_channels, len(celltypes))

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
    

class BasePairHead(nn.Module):
    """
    Predicts from the full base-pair resolution profile - NOT A DISTRIBUTION
    Calculated the profile and scalar from the predicted base-pair resolution
    Scalar calculated from the summing across the ocr region
    Profile is calculated from dividing the profile by the sum
    Bias (a probability distribution) is added to the profile just before the output
    """
    def __init__(self, in_channels:int, num_celltypes:int, ocr_start:int=375, ocr_end:int=625):
        """ 
        in_channels: the number of channels in the input data 
        seq_length: the length in base-pairs of the profile that you want to predict
        num_classes: the number of celltypes / output classes
        """
        super(BasePairHead, self).__init__()
        self.eps = 1e-8
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        kernel_size = 25 # TODO optimize - bp net used this bc 25bp is typical ChIP nexus width
        self.deconvolution = nn.Conv1d(in_channels, out_channels=num_celltypes, kernel_size=kernel_size, padding='same') # we need to do this for every cell type 

    def forward(self, x, bias):
        """
        x: the input data to this layer - expected output from model body
        bias: the bias associated with this sequence that x is derived from
            This is a probability distribution
        """
        # bias is currently a double tensor so we cast to a float - THIS COULD BE A SOURCE OF small ERROR, 
        bias = bias.float() 
        # get out to number of cellnumbers # of channels
        out = self.deconvolution(x)
        bp_pred = F.relu(out)

        # get the total counts of the OCR region
        ocr_region = bp_pred[..., self.ocr_start : self.ocr_end]
        scalar_pred = torch.sum(ocr_region, dim=-1)

        out = self.deconvolution(x)

        # get the probability distribution by dividing by sum accross all bp predictions
        profile_pred = (bp_pred + self.eps)  / torch.sum(bp_pred + self.eps) # TODO should I add an epsilon here?
        # correct for bias
        profile_pred = profile_pred + torch.unsqueeze(bias, dim=1)
        # perform a ReLU again to ensure no negative value
        profile_pred = F.relu(profile_pred)
        # now make it a distribution again by dividing by the sum --- again
        profile_pred = profile_pred / torch.sum(profile_pred)
        return profile_pred, scalar_pred

class MSELogs(nn.Module):
    """
    This loss function takes the mse of logs of vectors
    This is the MSE term used in BPNet
    (log(v_1 + 1) - log(v_2 + 1))^2
    """
    def __init__(self, reduction):
        super(MSELogs, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, label):
        log_prediction = torch.log(prediction + 1)
        log_label = torch.log(label + 1)
        return self.mse(log_prediction, log_label)



class CompositeLoss(nn.Module):
    """ 
    The loss function used in bpnet:
    a combination of cross entropy loss and 
    squared error of the log total number of reads  
    Returns the composite loss, the individual scalar loss, and the profile loss
    all meaned across the sequence, all sequences and celltypes.
    
    _lambda is the weight given to the profile loss function term (multinomial loss)
    """

    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super(CompositeLoss, self).__init__()
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.cross_entropy = CrossEntropyLoss(reduction)
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
    
    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        
        profile_error = self.cross_entropy(bp_counts, profile_prediction) 
        scalar_error = self.mse(total_count_prediction, total_counts)
        loss = _lambda * profile_error + scalar_error
        return loss, scalar_error, profile_error

class CompositeLossBalanced(nn.Module):
    """
    Loss function similar to Composite loss, but the scalar error and profile error are weighted with (1-lambda) and lambda respectivelys
    A combination of MSE logs and cross entropy loss

    ocr_only: boolean if the final dimension of the input should be sliced to conly consider the region ocr_start : ocr_end
    
    """
    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super(CompositeLossBalanced, self).__init__()
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.jsd = JSDLoss(reduction)
        self.cross_entropy = CrossEntropyLoss(reduction)
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
    
    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        assert _lambda <= 1 and _lambda >=0
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]            
        profile_error = self.cross_entropy(bp_counts, profile_prediction) 
        scalar_error = self.mse(total_count_prediction, total_counts)
        loss = _lambda * profile_error + (1-_lambda) * scalar_error
        return loss, scalar_error, profile_error

class CompositeLossBalancedJSD(nn.Module):
    """ 
    The loss function used in bpnet:
    a combination of cross entropy loss and 
    squared error of the log total number of reads  
    Returns the composite loss, the individual scalar loss, and the profile loss
    all meaned across the sequence, all sequences and celltypes.
    
    _lambda is the weight given to the profile loss function term (multinomial loss)
    """

    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super(CompositeLossBalancedJSD, self).__init__()
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.jsd = JSDLoss(reduction)
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
    
    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        
        profile_error = self.jsd(bp_counts, profile_prediction) 
        scalar_error = self.mse(total_count_prediction, total_counts)
        loss = _lambda * profile_error + (1-_lambda) * scalar_error
        return loss, scalar_error, profile_error
    
class JSDLoss(nn.Module): 
    """
    This loss funciton uses JSD
    See functions.py for details 
    """
    def __init__(self, reduction:str):
        """
        only supported reduction is 'mean'
        """
        super(JSDLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, profile_prediction, bp_counts):
        # THE JSD function does not take into account cell types correctly
        jsd = JSD(profile_prediction, bp_counts, self.reduction)
        return jsd

class CrossEntropyLoss(nn.Module):
    """
    Calculates the cross entropy 
    -Sum[ p(x) log(q(x)) ] across all predictions
    where p(x) is the target distribution and q(x) is the predicted probability distribution
    the given distributions must be probability distributions (i.e. from softmax)
    """
    def __init__(self, reduction:str):
        """
        the only reduction supported is 'mean' or none
        """
        super(CrossEntropyLoss, self).__init__()
        self.reduction=reduction
    
    def forward(self, p, q):
        """
        p is target
        q is prediction
        NOTE THIS IS DIFFERENT p and q THAN OTHER FUNCTIONS BE CAREFUL
        p and q will both be normalized to become probability distributions
        expected that the last dimension is the sequence length
        ex (N, num_celltypes, seq_len)
        """
        # for debugging
        eps = 1e-15
        q = q + eps
        # p, q = p + eps, q + eps # NOTE: One thing that I need to try is just adding epsilon to q and not p 
                                #       The one problem that comes up is when dividing by normp if normp is zero

        # confim that p and q are probability distributions
        normp, normq = p.sum(dim = -1)[...,None], q.sum(dim = -1)[...,None] # sum across sequence length
        normp[normp == 0 ] = 1  # avoid divide by zero
        # expected dimension: (N, 90, 1)

        # normalize the sequence 
        p = p/normp # expected dimension (N, 90, 1000) the sum gets broadcast across the sequence length
        q = q/normq
        # first we find log(q(x))
        # find element-wise p(x)*log(x) then sum across sequence length
        H = -1 * (p * q.log()).sum(dim = -1)[...,None]
        if (self.reduction == 'mean'):
            mean = torch.mean(H)
            if torch.isnan(mean):
                print("Cross entropy IS NAN")
                assert False
            return mean
        else :
            return H


class AlexJSD(nn.Module):
    def __init__(self, sum_axis = -1, norm_last = True, reduction = 'none', eps = 1e-8, include_mse = True, mse_ratio = 10., mean_size = 25):
        super(AlexJSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)
        self.mse = None
        if include_mse:
            self.mse = nn.MSELoss(reduction = reduction)
            self.mean_size = mean_size
            self.meanpool = None    
        self.mse_ratio = mse_ratio
        self.sum_axis = sum_axis
        self.norm_last = norm_last
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, p: torch.tensor, q: torch.tensor):
        if self.mse is not None:
            if self.mean_size is None:
                self.mean_size = p.size(dim = -1)
            if self.meanpool is None:
                self.meanpool = nn.AvgPool1d(self.mean_size, padding = int((p.size(dim = -1)%self.mean_size)/2), count_include_pad = False)
                self.l_out = int(np.ceil(p.size(dim = -1)/self.mean_size))
            pn = self.meanpool(p).repeat(1,1,self.mean_size)
            qn = self.meanpool(q).repeat(1,1,self.mean_size)
        
        # ONLY USE JSD if values are GREATER than ZERO. Make sure by using RELU, Sigmoid, or Softmax function
        #p = p-torch.min(p,dim =-1)[0].unsqueeze(dim=-1)
        #q = q-torch.min(q,dim =-1)[0].unsqueeze(dim=-1)
        p, q = p + self.eps, q + self.eps
        if self.norm_last:
            normp, normq = p.sum(dim = -1)[...,None], q.sum(dim = -1)[...,None]
            #normp[normp == 0] = 1.
            #normq[normq == 0] = 1.
            p = p/normp
            q = q/normq
        m = (0.5 * (p + q)).log()
        p = p.log()
        q = q.log()
        kl = 0.5 * (self.kl(m, p) + self.kl(m, q)) 
        if self.sum_axis is not None:
            klsize = kl.size()
            if self.sum_axis == -1:
                self.sum_axis = len(klsize) -1
            kl = kl.sum(dim = self.sum_axis)
            if self.reduction == 'none':
                kl = kl.unsqueeze(self.sum_axis).expand(klsize) #self.expand)
        if self.mse is not None:
            kl = kl + self.mse_ratio* self.mse(pn, qn)
        return kl

class ProfileBasedLoss(nn.Module):
    def __init__(self, ocr_begin:int=375, ocr_end:int=625):
        """
        Note: ocr_begin and ocr_end should match the regions
        covered by the observed total counts input in the forward
        method
        """
        super(ProfileBasedLoss, self).__init__()
        reduction = 'mean'
        self.cross_entropy = CrossEntropyLoss(reduction)
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
        self.eps = 1e-8
        self.ocr_begin = ocr_begin
        self.ocr_end = ocr_end
    
    def forward(self, obs_bp_counts, obs_total_counts, profile_prediction, 
                _lambda):
        """
        This function will calculate a scalar error based on the sum over the center
        of the sequence (center start and end are indicated by ocr_begin and ocr_end),
        and evaluate this using MSE.
        Calculates a profile distribution prediction of the OCR region of the profile_prediction
        and evaluates it using cross-entropy. 
        Forms a composite loss value of MSE + _lambda*cross entropy

        Arguments:
        obs_bp_counts: pytorch tensor of observed base pair counts - expected ~1000bp seqs
        obs_total_counts: pytorch tensor of observed total counts in OCR regions
        profile_prediciton: pytorch tensor of the predicted profile
            output from the model. Designed to work with models such as
            BPoh which only output a single profile. This profile
            should NOT be a probability distribution, just a pure
            prediction of the profile
            Asumes that the shape is (# samples, # celltypes, sequence len)
        ocr_begin: the index in the sequence in which the ocr is considered
            to begin
        ocr_end: the index in the sequence where the ocr ends
        """
        pred_ocr_profile = profile_prediction[..., self.ocr_begin : self.ocr_end]
        pred_scalar = torch.sum(pred_ocr_profile)
        # profile prediction
        pred_distribution = (pred_ocr_profile + self.eps)/ torch.sum(pred_ocr_profile)

        profile_error = self.cross_entropy(pred_distribution, obs_bp_counts[..., self.ocr_begin : self.ocr_end])
        scalar_error = self.mse(pred_scalar, obs_total_counts)
        loss = _lambda * profile_error + scalar_error
        return loss, scalar_error, profile_error

class CompositeLossMNLL(nn.Module):
    """ 
    The loss function used in bpnet:
    a combination of cross entropy loss and 
    squared error of the log total number of reads  
    Returns the composite loss, the individual scalar loss, and the profile loss
    all meaned across the sequence, all sequences and celltypes.

    CompositeLossMNLL = lambda * MNLL(profile) + (1-lambda) * MSE(scalar)
    
    _lambda is the weight given to the profile loss function term (multinomial loss)
    """

    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super(CompositeLossMNLL, self).__init__()
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.mnll = MNLLLoss(reduction=reduction)
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
    
    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts expected that the last dimension is the sequence length
            ex (N, num_celltypes, seq_len)
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        if _lambda > 1 or _lambda < 0:
            print("ERORR _lambda must be [0, 1]")
            return 
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        

        profile_error = self.mnll(profile_prediction, bp_counts, total_counts) 
        scalar_error = self.mse(total_count_prediction, total_counts)
        loss = _lambda * profile_error + (1-_lambda) * scalar_error
        return loss, scalar_error, profile_error
    
class MNLLLoss(nn.Module):
    """A loss function based on the multinomial negative log-likelihood.

    This loss function takes in a tensor of normalized  probabilities such
    that the sum of each row is equal to 1 (e.g. from a softmax) and
    an equal sized tensor of true counts and returns the probability of
    observing the true counts given the predicted probabilities under a
    multinomial distribution. Can accept tensors with 2 or more dimensions
    and averages over all except for the last axis, which is the number
    of categories.

    For optimization, this method ignores the -log(N!/[k! ...k_d!]) term because
    it is not affected by optimization. Meaning that the gradient 
    is zero:

    Multinomial negative log likelihood loss:
    Let [k_1! ... k_L!] = K 
    MNLL = -log(N!/K * PI[p_i^k_i)])
    = log(N!/K) - Sum[log(p_i^k_i)]
    = const - Sum[k_i * log(p_i)]
    = const - Sum[N * k_i / N *log(p_i)]   
    = const - N * Sum[q_i * log(p_i)]
    = const + N * CrossEntropyLoss()
     
    Parameters
    ----------
    ps: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories. 
        Must be normalized, representing probabilities

    true_counts: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples 
        representing observed counts

    total_counts: torch.tensor, shape=(n, ..., )
        A tensor with `n` examples 

    Returns
    -------
    loss: float
        The multinomial log likelihood loss of the true counts given the
        predicted probabilities, averaged over all examples and all other
        dimensions.

    """
    def __init__(self, reduction):
        super(MNLLLoss, self).__init__()
        self.reduction = reduction
        self.cross_entropy = CrossEntropyLoss(reduction='none')
    
    def forward(self, pred_bp_probs, obs_bp_counts, total_counts):
        cross_entropy = torch.squeeze(self.cross_entropy(obs_bp_counts, pred_bp_probs))
        out = total_counts * cross_entropy
        if self.reduction == 'mean':
            return out.mean()
        return out
    
    

class CompositeLossMSE(nn.Module):
    """
    This class computes MSELogs(profile_prediction*predicted_total_counts, actual bp counts)
    takes in reduction for compatability, but will always use mean
    """
    
    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super().__init__()
        self.reduction = 'mean'
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.mse_loss = MSELogs(reduction=reduction)

    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts expected that the last dimension is the sequence length
            ex (N, num_celltypes, seq_len)
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        if _lambda > 1 or _lambda < 0:
            print("ERORR _lambda must be [0, 1]")
            return 
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        
        # profile_prediction = profile_prediction.permute(0, 2, 1) # switch the last two dimensions so that the format is n, seq_len, n_celltypes
        # bp_counts = bp_counts.permute(0, 2, 1)
        norm_profile_prediction, norm_bp_counts = normalize(profile_prediction, bp_counts)

        pred_total_counts_broadcast = total_count_prediction.unsqueeze(-1).repeat(1, 1, 250)
        pred_bp_counts = norm_profile_prediction * pred_total_counts_broadcast

        profile_error = self.mse_loss(pred_bp_counts, bp_counts)
        scalar_error = torch.tensor(0)
        loss = profile_error
        
        return loss, scalar_error, profile_error


class MSEDoubleBP(nn.Module):
    """
    MSE(scalar) + MSE(total counts * pred_profile)

    ocr_only: boolean if the final dimension of the input should be sliced to conly consider the region ocr_start : ocr_end
    
    """
    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super(MSEDoubleBP, self).__init__()
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
    
    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        
        norm_profile_prediction, norm_bp_counts = normalize(profile_prediction, bp_counts)

        # print("norm_profile_prediction stats:")
        # print(f"Min: {norm_profile_prediction.min().item():.6f}")
        # print(f"Max: {norm_profile_prediction.max().item():.6f}")
        # print(f"Mean: {norm_profile_prediction.mean().item():.6f}")
        # print(f"Sum along last dim: {norm_profile_prediction.sum(dim=-1).mean().item():.6f}")

        # Reshape total_counts to [20, 90, 1] and broadcast it to [20, 90, 250]
        # total_counts_broadcast = total_counts.unsqueeze(-1).expand_as(norm_profile_prediction)
        pred_total_counts_broadcast = total_count_prediction.unsqueeze(-1).repeat(1, 1, 250)

        pred_bp_counts = norm_profile_prediction * pred_total_counts_broadcast
   
        # Check if the sum along the last dimension equals the total counts
        # sum_pred_bp_counts = torch.sum(pred_bp_counts, dim=-1)
        # print('Max difference:', torch.max(torch.abs(sum_pred_bp_counts - total_count_prediction)))
        # assert torch.allclose(sum_pred_bp_counts, total_count_prediction, atol=1e-6)

        profile_error = self.mse(pred_bp_counts, bp_counts) 
        scalar_error = self.mse(total_count_prediction, total_counts)
        loss = _lambda * profile_error + (1-_lambda) * scalar_error
        return loss, scalar_error, profile_error

class MSEDoubleProfile(nn.Module):
    """
    MSE(scalar) + MSE(total counts * pred_profile)

    ocr_only: boolean if the final dimension of the input should be sliced to conly consider the region ocr_start : ocr_end
    
    """
    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super(MSEDoubleProfile, self).__init__()
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.mse = MSELogs(reduction=reduction) # trying out the MSE loss of the logs of total counts
    
    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        
        norm_profile_prediction, norm_bp_counts = normalize(profile_prediction, bp_counts)
        profile_error = self.mse(norm_profile_prediction, norm_bp_counts) 
        scalar_error = self.mse(total_count_prediction, total_counts)
        loss = _lambda * profile_error + (1-_lambda) * scalar_error
        return loss, scalar_error, profile_error

class PoissonNLL(nn.Module):
    def __init__(self):
        super(PoissonNLL, self).__init__()

    def forward(self, predictions, targets):
        # Ensure predictions are positive (as they represent λ in Poisson distribution)
        predictions = torch.clamp(predictions, min=1e-10)  # Avoid log(0)

        # Compute the Poisson NLL for each element
        nll = - (targets * torch.log(predictions) - predictions)

        # Average over the cell types (cell_number dimension)
        nll_avg = torch.sum(nll, dim=-1)  # Average over cell_number

        return nll_avg  # Shape will be (batch_size, sequence_length)
    
class PoissonLoss(nn.Module):
    """
    This class computes Negative binomial loss of the bp count data as 
    predicted by predicted total counts * predicted probability profile 
    """
    
    def __init__(self, reduction, ocr_only:bool, ocr_start:int=375, ocr_end:int=625):
        super().__init__()
        self.reduction = reduction
        self.ocr_only = ocr_only
        self.ocr_start = ocr_start
        self.ocr_end = ocr_end
        self.poisson_nll = nn.PoissonNLLLoss(log_input=False, reduction='mean')

    def forward(self, bp_counts, total_counts, profile_prediction, total_count_prediction, _lambda):
        """
        bp_counts: the observed base-pair atac seq counts expected that the last dimension is the sequence length
            ex (N, num_celltypes, seq_len)
        total_counts: observed summed counts in the OCR region (ex middle 250 bp)
        profile_prediction: a probability distribution of the predicted profile 
            across the entire sequence length
        total_count_prediction: a scalar prediction of the sum of counts in OCR region
        lambda: the coefficient weight for cross entropy of profile prediction 
        """
       
        if self.ocr_only:
            bp_counts = bp_counts[:, :, self.ocr_start:self.ocr_end]
            profile_prediction = profile_prediction[:, :, self.ocr_start:self.ocr_end]
        
        norm_profile_prediction, norm_bp_counts = normalize(profile_prediction, bp_counts)
        seq_len = bp_counts.shape[-1]
        pred_total_counts_broadcast = total_count_prediction.unsqueeze(-1).repeat(1, 1, seq_len)
        pred_bp_counts = norm_profile_prediction * pred_total_counts_broadcast
        profile_error = self.poisson_nll(pred_bp_counts, bp_counts)
        if (self.reduction == 'mean'):
            profile_error = torch.mean(profile_error)
            if torch.isnan(profile_error):
                print("Poisson loss IS NAN")
                assert False
        else :
            print('unsupported reduction type')
            assert False
        
        scalar_error = torch.tensor(0)
        loss = profile_error
        
        return loss, scalar_error, profile_error

    
class Bin(nn.Module):
    """
    This class has one function: to bin 1 dimensional pytorch tensors
    into width 
    When the binned dimension of input x is not perfectly divisible by self.width
    the remained elements at the end of the array will be ignored
    dim: the dimension that will be binned
    """
    def __init__(self, width:int, dim:int):
        super(Bin, self).__init__()
        self.width = width
        self.dim = dim
    def forward(self, x):
        """x is the input that will binned. This function is designed for 1-dimensional tensors only"""
        # Unfold the tensor along the second dimension with a step size of self.width
        unfolded = x.unfold(self.dim, self.width, self.width)
        # Sum along the unfolded dimension
        binned = unfolded.sum(self.dim + 1)
        return binned

class AverageWrapper(torch.nn.Module):
    '''
    Wrapper for cnn get attributions for average of tracks
    
    Parameters
    ----------
    model : pytorch.nn.module
        The cnn_multi object
    tracks : list of lists
        tuples contain all the track indices that are combined to averages
    '''
    def __init__(self, model, tracks):
        super(AverageWrapper, self).__init__()
        self.model = model
        self.tracks = tracks

    
    def forward(self, input, bias):

        profile, scalar = self.model(input, bias)
        batch_size, _, seq_length = profile.size()
        num_track_groups = len(self.tracks)
        
        avg_profile = torch.zeros(batch_size, num_track_groups, seq_length, device=profile.device)
        avg_scalar = torch.zeros(batch_size, num_track_groups, device=scalar.device)
        
        for i, track_group in enumerate(self.tracks):
            track_group = torch.tensor(track_group, device=profile.device)
            avg_scalar[:, i] = torch.mean(scalar[:, track_group], dim=1)
            avg_profile[:, i, :] = torch.mean(profile[:, track_group, :], dim=1)
        
        return avg_profile, avg_scalar
    
    
    # def predict(self, X, device = None, enable_grad = False):
    #     if device is None:
    #         device = self.model.device
        
    #     profile, scalar = self.model(input, bias)
    #     predout = batched_predict(self, X, device = device, batchsize = self.model.batchsize, shift_sequence = self.model.shift_sequence, random_shift = self.model.random_shift, enable_grad = enable_grad)
    #     return predout


