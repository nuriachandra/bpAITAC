# bpAITAC

This code base is designed for creating and exploring deep learning models which predict ATAC-seq profiles and ATAC-seq total OCR counts from gene sequences. The current best functioning model is BPcm. The code for the model can be found in BPcm.py. The components of this model can be found in modules.py. 

## Requirements
The packages required to run bpAITAC are included in `environment.yml`. In addition you will need to install pytorch. For results reported in the publication we use `pytorch-cuda=12.4`. You can intall pytorch using pip from `https://pytorch.org/get-started/locally/` 

# Data
The input data for this model must be 'memmaped'. The already prepared data can be found in ```/data/nchand/ImmGen/mouse/BPprofiles1000/memmaped/complete_bias_corrected_normalized_3.7.23``` in the Chelan lab cluster. The models are designed to load in data based on an info.txt file that is located in the same folder as all of the memmory-mapped data files. 

The celltype data path is ```/data/nchand/ImmGen/mouse/BPprofiles1000/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy```

# Training

## Training bpAITAC

train.py is used for training. run_scripts/train_script.sh is an example of a script that can be used to run train.py

```python train.py info_file celltypes_file seq_len name model output_dir loss_fxn num_epochs lambda bias n_filters ocr_eval batch_size learning_rate
```
Arguments:
  1. info_file: path to the info.txt file in the folder produced by prep_data 
                that contains info on the memmaped files of all data 
  2. celltypes - path to a numpy file with the celltypes in it
  3. length of the sequences in base pairs (ex 1000)
  4. name of the version of the model that is being tested
  5. name of the torch.nn model class that you would like to train (Options are: BPnetRep, and CNN_0)
  6. name of the output directory 
  7. loss function name (options: PoissonNLLLoss, MSELoss, CompositeLoss) Best is CompositeLoss
  8. number of epochs
  9. _lambda weight on profile head
  10. boolean for bias. If true bias will be included
  11. n_filters number of filters in the body of BPnetRep if this body is being used
  12. ocr_eval: boolean indicating if only the ocr region: (middle 250 bp)
      should be evaluated when evaluating profile prediction in tran and validation
  13. batch size (should be ~20 for BPcm)
  14. learning rate

  Example

  ```python train.py info_file_path celltypes_file_path 1000 BP17_L-1_0.7 BPcm output_dir CompositeLoss 50 0.7 True 300 True 20 0.001
  ```

  ## Training the Tn5 bias model
  TODO 

## Loading the Model
Use the load_model function in load_model.py. 
The model type to load is specified manually inside of the function. 
Example:
```
load_model('/data/nchand/analysis/BP6_L-11/04-16-2023.21.51/best_model', n_celltypes=90, n_filters=300)
```


## Authors

* **Nuria Alina Chandra** 


## Acknowledgments

* Alexander Sasse 
* Sara Mostafavi


## Running remote server
on Chelan:
cd into file, and activate ai-tac
jupyter notebook --no-browser --port=8888

on current server:
ssh -N -f -L localhost:8888:localhost:8888 nchand@chelan.cs.washington.edu

### Using Hyak
to allocate 
 salloc -A mostafavilab -p gpu-a40 -N 1 -c 10 --gpus 1 --mem=80G --time=2:30:00