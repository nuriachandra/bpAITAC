from typing import List
import numpy as np
import pickle
import os
import sys
from prep_data_utils import quantile_norm_center, get_total_counts, get_total_ocr_counts, average_multi_track, average_single_track, get_lineage_cells
import argparse


"""
This file contains the methods used to get data ready to be run by bpAITAC models
and makes a memory mapping of all the data. It divides data into test and training sets based on chromosome number
Currently:  test_chr = ['chr11', 'chr16']   val_chr = ['chr12', 'chr15']   
makes memmory maps for the divided test train and validation data
and saves them in a folder called "memmaps"

Arguments:
1) onehot_npz: file path to npz file containing a file named 'seqfeatures' that 
  has the onehot encodings in it. and a 'genenames' file that 
  has the peak names in it
  ex) /data/nchand/mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz
2) atac_npz: filepath to npz file with file 'counts' with base-pair counts
  'names' with peak names, and 'celltypes' with the names of the 90
  celltypes represented in the atac file
  ex) /data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz
3) bias_npz: filepath to npz file with file 'counts' containing predicted tn5 cut bias 
  and file 'names' with peak names 
  ex) /data/nchand/ImmGen/mouse/BPprofiles1000/bias/CNNdilbiasfromcleanBACs_loglike.npz
5) path of pickled dictionary that maps peak names to chromosome name
6) directory where the output data will be stored and the memmaps folder will be created
7) if the data should be like in ai-tac True or False
8) file path to numpy array mask for the cell types if ai-tac
9) True if the bias is ahead of atac_counts by 2
10) --lineage-filepath defaults to None 
11) --selected-lineage defaults to None. If 'all' then a multi-task model with lineages will be selected. Otherwise should be a lineage name such as 'Stem&Prog' or 'B'
12) --cell-names-filepath defaults to None. Only needed if selected-lineage is specified. ex) cell_names = np.load("/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy")

"""

def train_test_split(test_chr:List[str], val_chr:List[str], 
                    names:np.ndarray, peaks_chr_dict:dict,
                    ) ->tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  This method divides the given numpy files into testing and training data 
  test_chr: list of strings of the chromosomes to be used as the test set ex) ['chr11']
  val_chr: list of strings of the chromosomes to be used as validation set ex) ['chr12', 'chr15']
  names: np array of peak names
  peaks_chr_dict a dictionary that maps each peak name to its chromosome name 

  returns a three boolean numpy arrays describing whether each peak/index is in
      train, test or validation set
      NOTE: This can only be used to splice other arrays that have the same peak sequence order as names   
  """
  # go through every peak and see if it is in training or validation or testing
  # make three bool[] arrays for train, val, and test
  # these represent whether the peak at each index is part of that set
  test = np.zeros(names.size, dtype=bool)
  val = np.zeros(names.size, dtype=bool)
  train = np.zeros(names.size, dtype=bool)
  for i in range(names.size):
    peak = names[i]
    if peaks_chr_dict[peak] in test_chr:
      test[i] = True
    elif peaks_chr_dict[peak] in val_chr:
      val[i] = True
    else:
      train[i] = True
  return train, test, val


def make_memmap(data:np.ndarray, path:str, info_file:str):
  """
  This method makes a memory map for a given numpy file 
    and saves it at the given path
  save filename, dtype and shape at given info_file path, 
    The information stored in the info_file is necessary so that
    memmap can be opened later 

  Arguments:
  data: the numpy array that will be converted to a memmap 
  path: a string of the file path that the memmap should be stored at
  info_file: the path of a txt file where the name, datatype and shape of each memmap array should be stored
  """
  fp = np.memmap(path, dtype=data.dtype, mode='w+', shape=data.shape)
  fp[:] = data[:]
  fp.flush
  with open(info_file, 'a') as f:
    f.write('%r\t%r\t%r\n' % (path, fp.dtype, fp.shape))
  return fp

def prep_data(genome_one_hot_npz:str, atac_npz:str, bias_npz:str, peak_chr_dict_path:str,
              data_storage_path:str, aitac_settings=False, cell_mask_path:str='', off_by_two=True, 
              ocr_only:bool=False, lineage_filepath:str=None, selected_lineage:str=None, cell_names_filepath:str=None, 
              ocr_start:int=375, ocr_end:int=625): 
  """
  This method splits data into train and test sets and converts data into numpy memmaps pytorch to train and validate a model
  it will create a "memmaps" folder within data_storage_path directory and save all output files there

  genome_one_hot: file path to an np file containing the one hot encoded genomic data divided into sequences
    it is okay for the sequences to be overlapping
  atac_bp_counts: file path to an np file containing the base-pair level data for sequences for each peak
  bias: file path to np file containing the learned bias at each peak's 1000 bp region (see Alex's code for how to do this)
  atac_names: file path to a np file containg the peak names associated with the above atac_counts
  peak_chr_dict: file path to a pickled dictionary that maps 
    atac-seq peak names to chromosome names
  data_storage_path: the directory that the output file folder should be placed int
  """
  # load the numpy arrays that you want to convert to memmaps 
  # Note: I also perform some data modifications here that are not necessary to convert the files to memmaps  
  if not os.path.exists(data_storage_path):
    os.makedirs(data_storage_path)

  print('ocr start and end', ocr_start, ocr_end)

  # sort the files so they are in same order
  onehot_enc, bp_counts, bias, names = sort_data(genome_one_hot_npz, atac_npz, bias_npz, off_by_two)

  ### Quantile normalize the bp-count data ###
  bp_counts, total_counts = quantile_norm_center(bp_counts, ocr_start=ocr_start, ocr_end=ocr_end)

  if selected_lineage is not None:
    cell_names = np.load(cell_names_filepath)
    lineage_names, lineage_cell_indices = get_lineage_cells(lineage_filepath, cell_names=cell_names)
    if selected_lineage.lower() == 'all':
      print('creating data for all lineages')
      bp_counts, total_counts = average_multi_track(lineage_cell_indices=lineage_cell_indices, scalar=total_counts, profile=bp_counts)
      # save the lineage names into a numpy array 
      np.save(os.path.join(data_storage_path, 'lineage_names'), lineage_names)
      print('saved lineage names at', os.path.join(data_storage_path, 'lineage_names'))
    else:
      print('creating data for selected lineage', selected_lineage)
      bp_counts, total_counts = average_single_track(selected_lineage_name=selected_lineage, lineage_names=lineage_names, lineage_cell_indices=lineage_cell_indices, scalar=total_counts, profile=bp_counts)
      # make an array of just the appropriate lineage name
      lineage = np.array([selected_lineage])
      np.save(os.path.join(data_storage_path, 'lineage'), lineage)
      print('saved lineage name at', os.path.join(data_storage_path, 'lineage'))
  print('shape of bp counts', bp_counts.shape, 'shape of total counts', total_counts.shape)

  # IF only OCR of ATAC-seq data should be used then just slice out
  # # the center ocr_start and end
  if (ocr_only):
    print("IN OCR ONLY SECTION")
    bias = bias[:, :, ocr_start:ocr_end]
    bp_counts = bp_counts[:, :, ocr_start:ocr_end]
    print("after ocr only correction shapes:", bias.shape, bp_counts.shape)
    

  peaks_chr_dict = pickle.load(open(peak_chr_dict_path, 'rb'))# maps peak names to chromosome names 

  ### SPLIT TRAIN TEST AND VAL ### 
  if not aitac_settings:
    test_chr = ['chr11', 'chr16']   # leavout chr11 and chr16 for testing ['chr11', 'chr16'] 
    val_chr = ['chr12', 'chr15']
  else: # aitac settings
    print("in else statement")
    test_chr = [None]   # for AI-TAC we are jus leaving out nothing
    val_chr = ['chr15']   # leavout chr15 and chr12 for validation ['chr12', 'chr15']-- for AI-TAC we are just leaving out ch15
    cell_mask = np.load(cell_mask_path)
    bp_counts = bp_counts[:, cell_mask, :]
    print("one success splice")
  train, test, val = train_test_split(test_chr, val_chr, names, peaks_chr_dict)



  ### CONVERT TO MEMMAP ###
  #Note: we convert to float so that they can be read as tensor.float by pytorch
  
  # find the path name for the memmap
  memmap_dir = data_storage_path + "/memmap"
  info_file = data_storage_path + "/memmap/info.txt"
  if not os.path.exists(memmap_dir):
    os.makedirs(memmap_dir)
  

  # make and save test memmap files 
  # go through names, bp_counts, total_counts, and onehot_enc 
  splicers = [train, test, val]
  types = ['train', 'test', 'val'] 
  for i in range(3):
    type_name = types[i]
    print("making " + type_name + " memmap")
    splicer = splicers[i]
    make_memmap(names[splicer], memmap_dir + "/" + type_name + ".names.dat", info_file)
    make_memmap(bp_counts[splicer].astype(np.float32),  memmap_dir + "/" + type_name + ".bp_counts.dat", info_file)
    make_memmap(total_counts[splicer].astype(np.float32),  memmap_dir + "/" + type_name + ".total_counts.dat", info_file)
    make_memmap(bias[splicer].astype(np.float32), memmap_dir + "/" + type_name + ".bias.dat", info_file)
    make_memmap(onehot_enc[splicer].astype(np.float32), memmap_dir + "/" + type_name + ".onehot.dat", info_file)





def sort_data(onehot_npz, atac_npz, bias_npz, off_by_two:bool):
  """
  This method takes npz files, sorts them to make
  sure that they all have the same peak ordering
  and then returns numpy files (all sorted by same peak ordering)

  returns onehot_encoding, atac_counts, bias_counts, peak_names

  onehot_npz: path to npz file containing a file named 'seqfeatures' that 
    has the onehot encodings in it. and a 'genenames' file that 
    has the peak names in it
    ex) /data/nchand/mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz
  atac_npz: path to npz file with file 'counts' with base-pair counts
    'names' with peak names, and 'celltypes' with the names of the 90
    celltypes represented in the atac file
    ex) /data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.npz
  bias_npz: path to npz file with file 'counts' containing predicted tn5 cut bias 
    and file 'names' with peak names 
    ex) /data/nchand/ImmGen/mouse/BPprofiles1000/bias/CNNdilbiasfromcleanBACs_loglike.npz
  """
  onehot_encoding = np.load(onehot_npz, allow_pickle=True)['seqfeatures'][0] # dim 0 = the onehot enc dim 1=nucleotide order
  onehot_names = np.load(onehot_npz, allow_pickle=True)['genenames']
  
  bias_counts = np.load(bias_npz, allow_pickle=True)['counts']
  print("SIZE OF BIAS COUNTS", bias_counts.shape)
  bias_names = np.load(bias_npz, allow_pickle=True)['names']
  
  atac_names = np.load(atac_npz, allow_pickle=True)['names']
  atac_counts = np.load(atac_npz, allow_pickle=True)['counts']
  
  # first confirm that the names are all the same
  # each peak in each of the three files is in the other three files 
  intersect1 = np.intersect1d(onehot_names, atac_names)
  onehot_atac_bias_intersect = np.intersect1d(intersect1, bias_names)
  if (len(onehot_atac_bias_intersect) < len(onehot_names) or 
      len(onehot_atac_bias_intersect) < len(atac_names) or
      len(onehot_atac_bias_intersect) < len(bias_names)):
    print("at least one of the data files contains less peaks")
    assert False
  else:
    print("all files contain the same peaks")

  # sort onehot and check that all elements are also in atac and bias 
  onehot_sort = np.argsort(onehot_names)
  onehot_encoding, onehot_names = onehot_encoding[onehot_sort], onehot_names[onehot_sort]
  
  # sort atac 
  atac_sort = np.argsort(atac_names)
  atac_counts, atac_names = atac_counts[atac_sort], atac_names[atac_sort]

  # sort bias
  bias_sort = np.argsort(bias_names)[np.isin(np.sort(bias_names), atac_names)]
  bias_counts, bias_names = bias_counts[bias_sort], bias_names[bias_sort]

  # check that all names are the same exact ordering
  if (np.array_equal(atac_names, onehot_names) and np.array_equal(bias_names, onehot_names)):
    print("names are now all in same order")
  else:
    print("Error: names are not in same order")
    assert False

  # if the bias is ahead of the bp counts by two
  if (off_by_two):
    print(bias_counts.shape)
    if bias_counts.shape[-1] == 250:
      bias_counts = bias_counts[:, :, 2:] # keep all peaks, and only the last 998 bp
      # Calculate the mean value of bias_counts
      mean_value = np.mean(bias_counts)
      # Pad using the mean value instead of zeros
      bias_counts = np.pad(bias_counts, pad_width=((0,0), (0,0), (0,2)), mode='constant', constant_values=mean_value)
    else:
      bias_counts = bias_counts[:, :, 2:] # keep all peaks, and only the last 998 bp
    
    atac_counts = atac_counts[:, :, :-2] # keep all peaks, all celltypes, the first 998 bp
    # This code assumes that the bias is correctly aligned with the sequences
    onehot_encoding = onehot_encoding[:, 2:, :] # Alex confirmed that this is the correct method

  return onehot_encoding, atac_counts, bias_counts, onehot_names

  """
  x, xnames = read(datafile1)
  y,ynames = read(datafile2)
  xsort = np.argsort(xnames)[np.isin(np.sort(xnames), ynames)] 
  x, xnames = x[xsort], xnames[xsort]
  ysort = np.argsort(ynames)[np.isin(np.sort(ynames), xnames)]
  y, ynames = y[ysort],ynames[ysort]
  """

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('required_args', nargs=9)
  parser.add_argument('--lineage-filepath', default=None)
  parser.add_argument('--selected-lineage', default=None)
  parser.add_argument('--cell-names-filepath', default=None)
  
  args = parser.parse_args()
  aitac_settings = args.required_args[5] == 'True'
  off_by_two = args.required_args[7] == 'True'
  ocr_only = args.required_args[8] == 'True'
  
  prep_data(*args.required_args[:5], aitac_settings,
            args.required_args[6], off_by_two, ocr_only,
            args.lineage_filepath, args.selected_lineage, args.cell_names_filepath)


# aitac_settings = False
# python prep_data.py /data/nchand/mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz 