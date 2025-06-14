# bpAITAC

This code base is designed for creating and exploring deep learning models which predict ATAC-seq profiles and ATAC-seq total OCR counts from gene sequences. The current best functioning model is BPcm. The code for the model can be found in BPcm.py. The components of this model can be found in modules.py. 

## Requirements
The packages required to run bpAITAC are included in `environment.yml`. In addition you may need to install pytorch. For results reported in the publication we use `pytorch-cuda=12.4`. You can intall pytorch using pip from `https://pytorch.org/get-started/locally/` 

# Data
Due to the large size of base-pair resultion data, the fullscale input data for this model must be 'memmaped'.  The models are designed to load in data based on a data config file. If the data is memmory mapped using the preprocessing code in `preprocessing/` a data config called info.txt will automatically be generated for you. 

# Training

## Training bpAITAC

train.py is used for training bpAITAC. Below is an example of how to train bpAITAC on the toy example data provided in `example_data/`

`python train.py --info_file example_data/data_config.yaml --celltypes_path  example_data/cell_names.npy --name bpAITAC_example --model_name bpAITAC --output_path example_data --seq_len 998 --memmaped_data False --num_epochs=10`


## Training the Tn5 bias model
We train the Tn5 bias model in an almost identical way to bpAITAC. However, a few model modifications are necessary to account for the single-track nature of training the Tn5 bias data. 


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

if the environment is not showing up as an option run
python -m ipykernel install --user --name=ai-tac
### Using Hyak
to allocate 
 salloc -A mostafavilab -p gpu-a40 -N 1 -c 10 --gpus 1 --mem=80G --time=2:30:00