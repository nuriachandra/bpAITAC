import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.modules import ProfileHead, ScalarHead
import math
import copy



class ConvNet(nn.Module):
    def __init__(self, num_filters:int, num_classes: int, seq_len: int, pool_size: int):
        """
        num_filters: int number of filters to be used in the convolution
        num_classes: the number of cell types we are predicting for
        seq_len: length of the genetic sequence
        pool_size: the size of the max pooling - should be set to the entire sequence length
        """
        super().__init__() # initialize the nn.Model, so we can use the nn.Model's methods
        in_channels = 4 # due to the number of nucleotide base-pairs
        self.seq_len = int(seq_len)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=25, padding='same'),
        )

        self.scalar_head = nn.Sequential(
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Flatten(start_dim=1, end_dim=2), 
            nn.Linear(in_features=num_filters, out_features=num_classes),
            nn.ReLU()
        )

        self.profile_head = ProfileHead(in_channels=num_filters, sequence_len=seq_len, num_classes=num_classes)
        

    def forward(self, input, bias):
        # run all layers on input data
        output = self.net(input)
        scalar = self.scalar_head(output) # + torch.sum(bias, dim=1, keepdim=True) # add the sequence bias summed over the sequence
        profile = self.profile_head(output, bias)
        return profile, scalar


# define the model loss
def pearson_loss(x, y):
    mx = torch.mean(x, dim=1, keepdim=True)
    my = torch.mean(y, dim=1, keepdim=True)
    xm, ym = x - mx, y - my

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = torch.sum(1 - cos(xm, ym))
    return loss


def train_model(train_loader, test_loader, model, device, criterion, optimizer, num_epochs, output_directory):
    total_step = len(train_loader)
    model.train()

    # open files to log error
    train_error = open(output_directory + "training_error.txt", "a")
    test_error = open(output_directory + "test_error.txt", "a")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device) # labels are the real ATAC-seq peak heights for 81 cell types (81 columns)

            # Forward pass
            outputs, act, idx = model(seqs)
            loss = criterion(outputs, labels)  # change input to
            running_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # save training loss to file
        epoch_loss = running_loss / len(train_loader.dataset)
        print("%s, %s" % (epoch, epoch_loss), file=train_error)

        # calculate test loss for epoch
        test_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (seqs, labels) in enumerate(test_loader):
                x = seqs.to(device)
                y = labels.to(device)
                outputs, act, idx = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()

        test_loss = test_loss / len(test_loader.dataset)

        # save outputs for epoch
        print("%s, %s" % (epoch, test_loss), file=test_error)

        if test_loss < best_loss_valid:
            best_loss_valid = test_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}'
                  .format(epoch + 1, best_loss_valid))

    train_error.close()
    test_error.close()

    model.load_state_dict(best_model_wts)
    return model, best_loss_valid


def test_model(test_loader, model, device):
    num_filters = model.conv_layer[0].out_channels
    predictions = torch.zeros(0, 81)
    max_activations = torch.zeros(0, num_filters)
    act_index = torch.zeros(0, num_filters)

    with torch.no_grad():
        model.eval()
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            pred, act, idx = model(seqs)  # what shape is the predictions returned
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
            max_activations = torch.cat((max_activations, act.type(torch.FloatTensor)), 0)
            act_index = torch.cat((act_index, idx.type(torch.FloatTensor)), 0)

    predictions = predictions.numpy()
    max_activations = max_activations.numpy()
    act_index = act_index.numpy()
    return predictions, max_activations, act_index


def get_motifs(data_loader, model, device):
    num_filters = model.conv_layer[0].out_channels
    activations = torch.zeros(0, num_filters, 251)
    predictions = torch.zeros(0, num_filters, 81)
    with torch.no_grad():
        model.eval()
        for seqs, labels in data_loader:
            seqs = seqs.to(device)
            pred, act, idx = model(seqs, num_filters)

            activations = torch.cat((activations, act.type(torch.FloatTensor)), 0)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)

    predictions = predictions.numpy()
    activations = activations.numpy()
    return activations, predictions
