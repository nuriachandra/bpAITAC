import torch.nn.functional as F 
import torch 
from torch import nn
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import stats 
import scipy




def JSD(P, Q, reduction='mean'):
    """
    Computes the Jensen-Shannon-divergence of two probability distributions 
    P: torch tensor, must be a probability distribution, the "prediction" 
    Q: torch tensor, must be a probability distribution, the "target" values
    P and Q must have no negative values
    P and Q are expected to have the sequence length dimension as their last dimension

    Input dimension expected (N, 90, 1000)
    """
    eps = 1e-15

    # check that inputs and targets sum to 1 
    p, q = P + eps, Q + eps
    normp, normq = p.sum(dim = -1)[...,None], q.sum(dim = -1)[...,None] # sum across sequence length
    # expected dimension: (N, 90, 1)

    # normalize the sequence 
    p = p/normp # expected dimension (N, 90, 1000) the sum gets broadcast across the sequence length
    q = q/normq

    m = (0.5 * (p + q)).log2()
    plog = p.log2()
    qlog = q.log2()

    # expected dimension (N, 90, 1000)

    # get pointwise KL divergence 
    # kl = 0.5 * (F.kl_div(m, p, reduction='none', log_target=True) + F.kl_div(m, q, reduction='none', log_target=True)) 
    kl = 0.5 * (p*(plog-m) + q*(qlog-m))

    # get the JSD over the SEQUENCES
    klsize = kl.size()
    # print(len(kl.size()))
    # sum over the last axis - the sequence length
    sum_axis = len(klsize) - 1
    kl = kl.sum(dim = sum_axis)

    # expected dimension (N, 90, 1)
    # outputs torch tensor with size (number of peaks, number of cell types)
    # returns the average sequence JSD throughout the batch, and throughout all cell types
    if reduction == 'mean':
        return torch.mean(kl)
    else:
        return kl


                    
def JSD_numpy(P, Q):
    """
    expects P and Q to be numpy arrays
    P is usually prediction Q is usually target
    of a single dimension 
    returns a single value for the jsd
    """
    eps = 1e-8 
    
    # check that inputs sum to 1 
    p, q = P + eps, Q + eps
    normp, normq = np.sum(p), np.sum(q) # sum across sequence length
    p, q = p/normp, q/normq

    distance = scipy.spatial.distance.jensenshannon(p, q, base=2)
    # based on scipy equations and wikipedia, divergence appears 
    # to be the square of distance 
    div = distance**2
    return div

def ocr_pearson_corr(x, y, dim:int=1, seq_len:int=1000, ocr_len:int=250):
    """
    The pearson correlation for the center 250 bp (the theoretical OCR)

    x and y must have seq len in their last dimension and batch_size in their first dimension
    specifically they must be of shape: (batch_size, num cell_types, seq_len)
    seq_len must be > ocr_len
    """
    print(x.size())
    print(y.size())
    
    # take off the extra on the sides of x and y
    flank = (seq_len - ocr_len) // 2
    x = x[:, :, flank : - flank]
    print(x.size())
    y = y[:, :, flank : - flank]
    print(y.size())
    out = pearson_corr(x, y, dim)
    # print(out.size())
    assert False
    return out

def pearson_corr(x, y, dim:int=1):
    """
    Returns the pearson correlations for all the peaks in a given batch

    x, y: tensors of size (batch_size, ... ), where first dimension is batch_size 
    dim is the dimension over which the correlation and mean be taken (ex mean over cell types, or mean over an entire sequence,
        which corresponds to correlation between value at each cell type, or value at each sequence position ) TODO check this is correct
        dim is automatically set to 1 which refers to finding the correlation between predicted and actual vectors x and y over all cell types
    """

    mx = torch.mean(x, dim=dim, keepdim=True)
    my = torch.mean(y, dim=dim, keepdim=True)
    xm, ym = x - mx, y - my

    corr_fxn = nn.CosineSimilarity(dim=dim, eps=1e-6) # similarity for each peak -> dim1
    # corr_sum = torch.sum(corr(xm, ym))
    corr = corr_fxn(xm, ym)
    # print(corr)
    return corr

def spearman_corr(x, y, dim:int=1):
   """Returns Spearman rank correlations by converting to ranks then using Pearson"""
   x_rank = x.argsort(dim=dim).argsort(dim=dim).float()
   y_rank = y.argsort(dim=dim).argsort(dim=dim).float()
   return pearson_corr(x_rank, y_rank, dim=dim)


def elementwise_pearson_corr(x:np.ndarray, y:np.ndarray, dim:int=-1, reduction='none'):
    """
    This method computes the pearson correlation for numnpy arrays with
    2 dimensions or more
    pearson correlation = cov(X,Y)/(std.dev(X)*std.dev(Y))
    pearson correlation coefficient approximation of a sample: 
        sum((x-mean(x))*(y-mean(y))) / (l2-norm(x)*l2-norm(y))
        sum and mean are computed across the given 'dim' axis
    This method uses the above equation for approximating cor. coefficient from a sample

    Note: currently this function returns nan for vectors with norm=0
        due to divide by zero. However, this situation does not commonly appear in 
        the observed data. TODO resolve this issue
    """

    mx = np.mean(x, axis=dim, keepdims=True)
    my = np.mean(y, axis=dim, keepdims=True)
    xm, ym = x - mx, y - my

    nx, ny = np.sqrt(np.sum(xm**2, axis = dim)), np.sqrt(np.sum(ym**2, axis = dim)) # calculating l2norm 
    nxy = nx*ny
    xmym = np.sum(xm*ym, axis = dim)
    
    # handling case when norm gets close to zero
    xmym = np.where(nxy/np.amax(np.array([nx,ny]),axis = 0) < 1e-8, -1, xmym)
    nxy = np.where(nxy/np.amax(np.array([nx,ny]),axis = 0) < 1e-8, 1, nxy)
    corout = xmym/nxy
    if reduction == 'nanmean':
        corout = np.nanmean(corout)
    return np.around(corout, 4)

def softmax_numpy(x):
    """Compute softmax values for each sets of scores in x. where x is a numpy array"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# def pearson_corr_bps(x, y):
#     """
#     Returns the sum of pearson correlations for all the OCRs in a given batch
#     Finds the average correlation between the predicted base-pair counts and the true
#     base pair counts over all cell types for a specific OCR. Then sums these values

#     x, y: tensors of size (batch_size, seq_length, num_cells). There is total count prediction per cell types
#     """
#     mx = torch.mean(x, dim=1, keepdim=True) # find average across the sequence
#     my = torch.mean(y, dim=1, keepdim=True)

#     xm, ym = x - mx, y - my

#     corr = nn.CosineSimilarity(dim=1)

def normalize(p, q):
    # profile_prediction = profile_prediction.permute(0, 2, 1) # switch the last two dimensions so that the format is n, seq_len, n_celltypes
    # bp_counts = bp_counts.permute(0, 2, 1)

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

    return p, q


def coefficient_of_variation(data, axis=1):
    # Calculate standard deviation along axis 1
    std_dev = np.std(data, axis=axis)
    
    # Calculate mean along axis 1
    mean = np.mean(data, axis=axis)
    
    # Calculate coefficient of variation
    cv = (std_dev / mean)
    return cv