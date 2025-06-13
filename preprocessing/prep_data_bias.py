import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

from prep_data_utils import get_total_counts, quantile_norm

def load_validation_names(validation_names_path):
    """
    Load validation names from a text file. Names can be space-separated on a single line
    or on separate lines.
    
    Args:
        validation_names_path (str): Path to text file containing validation names
        
    Returns:
        set: Set of validation names
    """
    if not Path(validation_names_path).exists():
        print(f"Error: Validation names file not found: {validation_names_path}")
        sys.exit(1)
        
    with open(validation_names_path, 'r') as f:
        content = f.read()
        # Split on whitespace to handle both space-separated and newline-separated names
        validation_names = set(name for name in content.split() if name)
        
    print(f"Loaded {len(validation_names)} validation names")
    return validation_names


def split_data(onehot_counts_matching, total_counts, bias, counts_quantile_normalized, 
               names_matching, validation_names=None, test_size=0.1, random_state=42):
    """
    Split all arrays into training and validation sets using either specified validation names
    or random split.
    
    Args:
        onehot_counts_matching: numpy array of onehot encodings
        total_counts: numpy array of total counts
        bias: numpy array of bias values
        counts_quantile_normalized: numpy array of normalized counts
        names_matching: list of matching names
        validation_names: set of names to use for validation (optional)
        test_size: fraction of data to use for validation (default: 0.2)
        random_state: random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_onehot, val_onehot, 
                train_total_counts, val_total_counts,
                train_bias, val_bias, 
                train_counts_norm, val_counts_norm,
                train_names, val_names)
    """
    if validation_names:
        # Convert names_matching to numpy array if it isn't already
        names_matching = np.array(names_matching)
        
        # Create boolean mask for validation set
        val_mask = np.array([name in validation_names for name in names_matching])
        train_mask = ~val_mask
        
        # Split using boolean masks
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
    else:
        # Use random split if no validation names provided
        indices = np.arange(len(total_counts))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state
        )
    
    # Split each array using the indices
    train_onehot = onehot_counts_matching[train_indices]
    val_onehot = onehot_counts_matching[val_indices]
    
    train_total_counts = total_counts[train_indices]
    val_total_counts = total_counts[val_indices]
    
    train_bias = bias[train_indices]
    val_bias = bias[val_indices]
    
    train_counts_norm = counts_quantile_normalized[train_indices]
    val_counts_norm = counts_quantile_normalized[val_indices]
    
    # Split the names using the same indices
    names_matching = np.array(names_matching)
    train_names = names_matching[train_indices]
    val_names = names_matching[val_indices]
    
    # Print split information
    print(f"\nData split summary:")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    if validation_names:
        print(f"Split method: Using provided validation names")
        print(f"Found {len(val_indices)} matching validation names")
    else:
        print(f"Split method: Random split (test_size={test_size})")
    
    return (train_onehot, val_onehot,
            train_total_counts, val_total_counts,
            train_bias, val_bias,
            train_counts_norm, val_counts_norm,
            train_names, val_names)




# def split_data(onehot_counts_matching, total_counts, bias, counts_quantile_normalized, 
#                names_matching, test_size=0.1, random_state=42):
#     """
#     Split all arrays into training and validation sets using the same indices.
    
#     Args:
#         onehot_counts_matching: numpy array of onehot encodings
#         total_counts: numpy array of total counts
#         bias: numpy array of bias values
#         counts_quantile_normalized: numpy array of normalized counts
#         names_matching: list of matching names
#         test_size: fraction of data to use for validation (default: 0.2)
#         random_state: random seed for reproducibility (default: 42)
    
#     Returns:
#         tuple: (train_onehot, val_onehot, 
#                 train_total_counts, val_total_counts,
#                 train_bias, val_bias, 
#                 train_counts_norm, val_counts_norm,
#                 train_names, val_names)
#     """
#     # Generate indices for the split
#     indices = np.arange(len(total_counts))
    
#     # Split indices
#     train_indices, val_indices = train_test_split(
#         indices,
#         test_size=test_size,
#         random_state=random_state
#     )
    
#     # Split each array using the same indices
#     train_onehot = onehot_counts_matching[train_indices]
#     val_onehot = onehot_counts_matching[val_indices]
    
#     train_total_counts = total_counts[train_indices]
#     val_total_counts = total_counts[val_indices]
    
#     train_bias = bias[train_indices]
#     val_bias = bias[val_indices]
    
#     train_counts_norm = counts_quantile_normalized[train_indices]
#     val_counts_norm = counts_quantile_normalized[val_indices]
    
#     # Split the names using the same indices
#     names_matching = np.array(names_matching)  # Convert to numpy array for indexing
#     train_names = names_matching[train_indices]
#     val_names = names_matching[val_indices]
    
#     # Print split information
#     print(f"\nData split summary:")
#     print(f"Training samples: {len(train_indices)}")
#     print(f"Validation samples: {len(val_indices)}")
    
#     return (train_onehot, val_onehot,
#             train_total_counts, val_total_counts,
#             train_bias, val_bias,
#             train_counts_norm, val_counts_norm,
#             train_names, val_names)


def sort_onehot_by_count_names(names_onehot, onehot, names_counts):
    """
    Extract and sort onehot encodings that match the names in names_counts.
    Only returns onehot encodings for names that appear in names_counts.
    
    Args:
        names_onehot (numpy.ndarray): Array of gene names from onehot data
        onehot (numpy.ndarray): Array of onehot encodings
        names_counts (numpy.ndarray): Array of peak names from counts data
    
    Returns:
        tuple: (matching_onehot, matched_names)
            - matching_onehot: Subset of onehot encodings corresponding to names in names_counts
            - matched_names: List of names that were matched, in the same order
    """
    # Create a dictionary mapping names to indices for quick lookup
    name_to_idx = {name: idx for idx, name in enumerate(names_onehot)}
    
    # Get indices of names_counts that exist in onehot data
    matching_indices = []
    matched_names = []
    
    for i, name in enumerate(names_counts):
        if name in name_to_idx:
            matching_indices.append(name_to_idx[name])
            matched_names.append(name)
    
    # Extract only the matching onehot encodings
    matching_onehot = onehot[matching_indices]
    
    print(f"Found {len(matching_indices)} matching names out of {len(names_counts)} names in counts data")
    
    return matching_onehot, matched_names

def filter_low_count_regions(onehot_counts_matching, total_counts, counts_quantile_normalized, 
                           bias, names_matching, min_counts=10):
    """
    Filter out regions with less than specified minimum counts.
    
    Args:
        onehot_counts_matching: numpy array of onehot encodings
        total_counts: numpy array of total counts
        counts_quantile_normalized: numpy array of normalized counts
        bias: numpy array of bias values
        names_matching: array of matching names
        min_counts: minimum count threshold (default: 10)
        
    Returns:
        tuple: (filtered_onehot, filtered_total_counts, filtered_counts_norm,
                filtered_bias, filtered_names)
    """
    # Create mask for regions with >= min_counts
    mask = np.squeeze(total_counts >= min_counts)
    print('mask shape', mask.shape)
    print('names_matching', np.array(names_matching).shape)
    
    # Apply mask to all arrays
    filtered_onehot = onehot_counts_matching[mask]
    filtered_total_counts = total_counts[mask]
    filtered_counts_norm = counts_quantile_normalized[mask]
    filtered_bias = bias[mask]
    filtered_names = np.array(names_matching)[mask]
    
    # Print filtering summary
    total_regions = len(total_counts)
    kept_regions = mask.sum()
    removed_regions = total_regions - kept_regions
    
    print(f"\nFiltering Summary:")
    print(f"Total regions before filtering: {total_regions}")
    print(f"Regions with >= {min_counts} counts: {kept_regions}")
    print(f"Regions removed: {removed_regions}")
    print(f"Percentage kept: {(kept_regions/total_regions)*100:.1f}%")
    
    return (filtered_onehot, filtered_total_counts, filtered_counts_norm,
            filtered_bias, filtered_names)

def save_split_data(output_dir, train_onehot, val_onehot,
                   train_total_counts, val_total_counts,
                   train_bias, val_bias,
                   train_counts_norm, val_counts_norm,
                   train_names, val_names):
    """
    Save all split arrays as separate .npy files in the specified output directory.
    
    Args:
        output_dir (str or Path): Directory where files should be saved
        train_* and val_* arrays: The split numpy arrays to save
        train_names and val_names: The split name arrays to save
    """
    output_dir = Path(output_dir)
    
    # Save training data
    np.save(output_dir / 'train_onehot.npy', train_onehot)
    np.save(output_dir / 'train_total_counts.npy', train_total_counts)
    np.save(output_dir / 'train_bias.npy', train_bias)
    np.save(output_dir / 'train_counts_norm.npy', train_counts_norm)
    np.save(output_dir / 'train_names.npy', train_names)
    
    # Save validation data
    np.save(output_dir / 'val_onehot.npy', val_onehot)
    np.save(output_dir / 'val_total_counts.npy', val_total_counts)
    np.save(output_dir / 'val_bias.npy', val_bias)
    np.save(output_dir / 'val_counts_norm.npy', val_counts_norm)
    np.save(output_dir / 'val_names.npy', val_names)


def load_and_validate_npz(onehot_path, counts_path):
    """
    Load and validate NPZ files containing genomic data.
    
    Args:
        onehot_path (str): Path to NPZ file containing genenames and seqfeatures
        counts_path (str): Path to NPZ file containing names and counts
        
    Returns:
        tuple: (genenames, seqfeatures, peak_names, counts)
    """
    try:
        # Load onehot NPZ file
        onehot_data = np.load(onehot_path, allow_pickle=True)
        print(onehot_data.files)
        names_onehot = onehot_data['genenames']
        onehot, bases = onehot_data['seqfeatures']
        print('onehot', onehot.shape)
        
        # Load counts NPZ file
        counts_data = np.load(counts_path)
        names_counts = counts_data['names']
        counts = counts_data['counts']
        
        # Validate data
        if len(names_onehot) != len(onehot):
            print('len names_onehot', len(names_onehot), 'len onehot', print(len(onehot)))
            raise ValueError("Mismatch between number of genenames and seqfeatures")
        
        if len(names_counts) != counts.shape[0]:
            raise ValueError("Mismatch between number of peak names and counts")
        
        return names_onehot, onehot, names_counts, counts
        
    except KeyError as e:
        print(f"Error: Missing required dataset in NPZ file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading NPZ files: {e}")
        sys.exit(1)

def save_config_yaml(output_dir):
    """
    Save a YAML configuration file with paths to all the data files.
    
    Args:
        output_dir (str or Path): Directory where data files are saved
    """
    import yaml
    
    output_dir = Path(output_dir)
    config_path = output_dir / 'bias_config.yaml'
    
    config = {
        'base_path': str(output_dir),
        'train_bias': str(output_dir / 'train_bias.npy'),
        'train_onehot': str(output_dir / 'train_onehot.npy'),
        'train_bp_counts': str(output_dir / 'train_counts_norm.npy'),
        'train_total_counts': str(output_dir / 'train_total_counts.npy'),
        'train_names': str(output_dir / 'train_names.npy'),
        'val_bias': str(output_dir / 'val_bias.npy'),
        'val_onehot': str(output_dir / 'val_onehot.npy'),
        'val_bp_counts': str(output_dir / 'val_counts_norm.npy'),
        'val_total_counts': str(output_dir / 'val_total_counts.npy'),
        'val_names': str(output_dir / 'val_names.npy'),
        'test_bias': str(output_dir / 'val_bias.npy'),
        'test_onehot': str(output_dir / 'val_onehot.npy'),
        'test_bp_counts': str(output_dir / 'val_counts_norm.npy'),
        'test_total_counts': str(output_dir / 'val_total_counts.npy'),
        'test_names': str(output_dir / 'val_names.npy')
    }
    
    # Add comments to the YAML string
    yaml_str = "# Training data paths\n"
    yaml_str += yaml.dump(config, default_flow_style=False)
    
    with open(config_path, 'w') as f:
        f.write(yaml_str)
    
    print(f"\nSaved configuration to {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Process genomic data from NPZ files')
    parser.add_argument('--onehot_npz', type=str, required=True,
                        help='Path to NPZ file containing genenames and seqfeatures files')
    parser.add_argument('--counts_npz', type=str, required=True,
                        help='Path to NPZ file containing files named names and counts')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save processed data')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='If set, only save first 2000 elements of arrays')
    parser.add_argument('--validation_names', type=str, default=None,
                        help='txt file. If given, should contain the names of peaks that should be put in the validation set. Not the training set')
    
    args = parser.parse_args()

    MIN_COUNTS=10
    
    # Validate input files exist
    if not Path(args.onehot_npz).exists():
        print(f"Error: One-hot NPZ file not found: {args.onehot_npz}")
        sys.exit(1)
    if not Path(args.counts_npz).exists():
        print(f"Error: Counts NPZ file not found: {args.counts_npz}")
        sys.exit(1)
    
    # Load validation names if provided
    validation_names = None
    print('arg val names', args.validation_names)
    if args.validation_names:
        if not Path(args.validation_names).exists():
            print(f"Error: Validation names file not found: {args.validation_names}")
            sys.exit(1)
        validation_names = load_validation_names(args.validation_names)
        print(f"Loaded {len(validation_names)} validation names")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and validate data
    names_onehot, onehot, names_counts, counts = load_and_validate_npz(
        args.onehot_npz, args.counts_npz
    )

    # Print summary statistics
    print("\nData Summary:")
    print(f"Number of genes: {len(names_onehot)}")
    print(f"Sequence features shape: {onehot.shape}")
    print(f"Number of peaks: {len(names_counts)}")
    print(f"Counts shape: {counts.shape}")
    

    onehot_counts_matching, names_matching = sort_onehot_by_count_names(
        names_onehot=names_onehot, 
        onehot=onehot, 
        names_counts=names_counts
    )

    # If counts are longer than 250 bp, take the middle 250 bp of counts in dimension 2
    if counts.shape[2] > 250:
        start_idx = (counts.shape[2] - 250) // 2
        counts = counts[:, :, start_idx:start_idx + 250]

    # If onehot is longer than 1000 bp in dimension 1, take the middle 1000
    if onehot_counts_matching.shape[1] > 1000:
        start_idx = (onehot_counts_matching.shape[1] - 1000) // 2
        onehot_counts_matching = onehot_counts_matching[:, start_idx:start_idx + 1000, :]


    print("\nAfter sorting and trimming Summary:")
    print(f"Onehot: {onehot_counts_matching.shape}")
    print(f"Number of peaks: {len(counts)}")
    print(f"Counts shape: {counts.shape}")

    total_counts = get_total_counts(counts)
    bias = np.zeros_like(counts)
    # counts_quantile_normalized = quantile_norm(counts) don't need to quantile normalize just one track

    # now I have to filter all of the regions with <10 counts
    # remove all the regions with <10 counts from onehot_counts_matching, total_counts, counts_quantile_normalized, bias, names_matching
    # Filter low count regions
    (filtered_onehot, filtered_total_counts, 
     filtered_counts_norm, filtered_bias, 
     filtered_names) = filter_low_count_regions(
        onehot_counts_matching=onehot_counts_matching,
        total_counts=total_counts,
        counts_quantile_normalized=counts,
        bias=bias,
        names_matching=names_matching,
        min_counts=MIN_COUNTS
    )

    # Split the filtered data

    (train_onehot, val_onehot,
     train_total_counts, val_total_counts,
     train_bias, val_bias,
     train_counts_norm, val_counts_norm,
     train_names, val_names) = split_data(
        onehot_counts_matching=filtered_onehot,
        total_counts=filtered_total_counts,
        bias=filtered_bias,
        counts_quantile_normalized=filtered_counts_norm,
        names_matching=filtered_names,
        validation_names=validation_names
    )

    # If sampling is enabled, take only first 2000 elements
    if args.sample:
        sample_size = min(2000, len(train_names))  # Ensure we don't exceed array length
        train_onehot = train_onehot[:sample_size]
        train_total_counts = train_total_counts[:sample_size]
        train_bias = train_bias[:sample_size]
        train_counts_norm = train_counts_norm[:sample_size]
        train_names = train_names[:sample_size]
        
        val_size = min(2000, len(val_names))  # Ensure we don't exceed array length
        val_onehot = val_onehot[:val_size]
        val_total_counts = val_total_counts[:val_size]
        val_bias = val_bias[:val_size]
        val_counts_norm = val_counts_norm[:val_size]
        val_names = val_names[:val_size]
        
        print(f"\nSampling enabled. Using {sample_size} training samples and {val_size} validation samples.")
    
    save_split_data(output_dir,
                   train_onehot, val_onehot,
                   train_total_counts, val_total_counts,
                   train_bias, val_bias,
                   train_counts_norm, val_counts_norm,
                   train_names, val_names)

    save_config_yaml(output_dir)
    
if __name__ == "__main__":
    main()


# example usage to prep the bias counts from the closed region counts 
# python prep_data_bias.py --onehot_npz /data/nchand/mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz --counts_npz /data/mostafavilab/ImgenATAC/ImmGen-log10pvaluesgt0.25_counts250bp_combined.npz --output_dir /data/nchand/ImmGen/mouse/bias/tn5_bias_counts250bp_combined_filtered_11.2.24 
# python prep_data_bias.py --onehot_npz /data/nchand/mm10/mm10ImmGenATAC1219.peak_matched1000bp_onehot-ACGT_alignleft.npz --counts_npz /data/mostafavilab/ImgenATAC/ImmGen-log10pvaluesgt0.25_counts250bp_combined.npz --output_dir /data/nchand/ImmGen/mouse/bias/tn5_bias_counts250bp_combined_filtered_11.2.24   --sample

# with the protein free dna
# python prep_data_bias.py --onehot_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/hg38selectedBACs_gt6avgcount_N500000bp1200_onehot-ACGT_alignleft.npz  --counts_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/selectedBACs_gt6avgcount_N500000bp1200_counts.npz --output_dir /data/nchand/ImmGen/mouse/bias/selectedBACs_gt6avgcount_N500000bp_filtered_1.2.25  --validation_names /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/selectedBACs_gt6avgcount_N500000bp1200_testset.txt
# python prep_data_bias.py --onehot_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/hg38selectedBACs_gt6avgcount_N500000bp1200_onehot-ACGT_alignleft.npz  --counts_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/selectedBACs_gt6avgcount_N500000bp1200_counts.npz --output_dir /data/nchand/ImmGen/mouse/bias/selectedBACs_gt6avgcount_N500000bp_filtered_11.2.24  
# python prep_data_bias.py --onehot_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/hg38selectedBACs_gt6avgcount_N500000bp1200_onehot-ACGT_alignleft.npz  --counts_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/selectedBACs_gt6avgcount_N500000bp1200_counts.npz --output_dir /data/nchand/ImmGen/mouse/bias/selectedBACs_gt6avgcount_N500000bp_filtered_11.2.24_sample --sample 

# with protein free 250 
# python prep_data_bias.py --onehot_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/hg38selectedBACs_N500000bp250_onehot-ACGT_alignleft.npz --counts_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/selectedBACs_N500000bp250_counts.npz --output_dir /data/nchand/ImmGen/mouse/bias/selectedBACs_gt6avgcount_N500000bp250_11.2.24 
# python prep_data_bias.py --onehot_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/hg38selectedBACs_N500000bp250_onehot-ACGT_alignleft.npz --counts_npz /data/mostafavilab/ImgenATAC/Tn5biasBuenrostro2022/selectedBACs_N500000bp250_counts.npz --output_dir /data/nchand/ImmGen/mouse/bias/selectedBACs_gt6avgcount_N500000bp250_11.2.24_sample --sample