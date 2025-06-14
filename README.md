# bpAITAC

This code base is designed for creating and exploring deep learning models which predict ATAC-seq profiles and ATAC-seq total OCR counts from gene sequences. We present bpAITAC, a new SOTA model for immunological ATAC-seq data modeling using base-pair resolution data. 

bioRxiv: https://www.biorxiv.org/content/10.1101/2025.01.24.634804v1 

## Requirements
The packages required to run bpAITAC are included in `environment.yml`. In addition you may need to install pytorch. For results reported in the publication we use `pytorch-cuda=12.4`. You can intall pytorch using pip from `https://pytorch.org/get-started/locally/` 

## Data
Due to the large size of base-pair resultion data, the fullscale input data for this model must be 'memmaped'.  The models are designed to load in data based on a data config file. If the data is memmory mapped using the preprocessing code in `preprocessing/` a data config called info.txt will automatically be generated for you. 

You can see examples of how to run the data preprocessing code in `preprocessing/prep_data.sh` 

## Training

### Training bpAITAC

train.py is used for training bpAITAC. Below is an example of how to train bpAITAC on the toy example data provided in `example_data/`

`python train.py --info_file example_data/data_config.yaml --celltypes_path  example_data/cell_names.npy --name bpAITAC_example --model_name bpAITAC --output_path example_data --seq_len 998 --memmaped_data False --num_epochs=10`


### Training the Tn5 bias model
We train the Tn5 bias model in an almost identical way to bpAITAC. However, a few model modifications are necessary to account for the single-track nature of training the Tn5 bias data. 

1. To prep the data to train the bias model, use `preprocessing/prep_data_bias.py`. Then make a config file from the data. This will need to have the same format as `example_data/data_config.yaml`
2. To train the Tn5 bias model, use the `train.py` script with the `--model_name BPbi` 
3. To compute the logits from the Tn5 bias model which can then be fed into training bpAITAC, use `preprocessing/prep_get_bias_logits.py`



## Authors
* **Nuria Alina Chandra** 


### Acknowledgments
* Alexander Sasse 
* Sara Mostafavi
* Yan Hu
* Jason D. Buenrostro

