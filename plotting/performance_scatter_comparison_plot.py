import numpy as np
from plot_results import scatter_heat
import matplotlib.pyplot as plt


def convert_to_txt(analysis_file_path, numpy_file_name, output_path):
    data = np.load(analysis_file_path)[numpy_file_name]
    peak_names =  np.load('/data/nchand/analysis/BPcm/BP17_setup_data/val_peak_names.npy')
    # Convert peak_names to a format compatible with savetxt
    combined = np.vstack((peak_names, data))
    combined = np.transpose(combined)
    print("combined shape", combined.shape)
    np.savetxt(output_path, combined, fmt='%s', delimiter='\t')

def get_data(analysis_path1, analysis_path2, analysis_file_name):
    array1 = np.load(analysis_path1)[analysis_file_name]
    array2 = np.load(analysis_path2)[analysis_file_name]
    return array1, array2

def scatter_plot_by_celltype(celltypes, x_data:np.ndarray, y_data:np.ndarray, x_label:str, y_label:str, outpath:str):
    select_cells = ['B.FrE.BM', 'B.mem.Sp', 'T.4.Nve.Fem.Sp', 'GN.Thio.PC']
    mask = np.isin(celltypes, select_cells)
    cell_idx = np.where(mask)[0]
    celltypes = celltypes[cell_idx]
    fig, ax = plt.subplots()
    if (x_data.shape[0] == y_data.shape[0] & x_data.shape[1] == y_data.shape[1]):
        print("ERROR!!! shapes not the same", x_data.shape, y_data.shape)
    x_data = x_data[:, cell_idx]
    y_data = y_data[:, cell_idx]
    n_celltypes = cell_idx.shape[0]
    print("number of celltypes", n_celltypes)
    colors = plt.cm.viridis(np.linspace(0, 1, n_celltypes))

    # Loop through each cell type
    for i in range(n_celltypes):
        # Extract data for the current cell type in both conditions
        x = x_data[:, i]
        y = y_data[:, i]
        
        # Plot a scatter plot for the current cell type with a unique color
        ax.scatter(x, y, label=celltypes[i], color=colors[i], alpha=0.7)

    # Set labels and a legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # Show the plot
    plt.savefig(outpath + '.jpg', format='jpg')

    plt.close()




def plot_scalar_correlation():
    lambda0, lambdap7 = get_data("/data/nchand/analysis/BPcm/BP17_L0_0/09-22-2023.08.38/analysis.npz", "/data/nchand/analysis/BPcm/BP17_L-1_7/05-26-2023.14.26/analysis.npz", 'scalar_corr')
    scatter_heat(lambda0, lambdap7, "$\lambda$=0", "$\lambda$=0.7", "/data/nchand/analysis/BPcm/BP17lambda0vs0.7_scalar_correlation_2")
    bin1, bin5 = get_data("/data/nchand/analysis/BPcm/BP33_1_L0_0.9/scalar_corr_avg.tsv", "/data/nchand/analysis/BPcm/BP33_5_L0_0.9/scalar_corr_avg.tsv", 'scalar_corr')
    scatter_heat(bin1, bin5, "No Binning", "Bin Size = 5", "/data/nchand/analysis/BPcm/BP33bin1vsbin5_scalar_correlation_1")

def plot_bp_correlation():
    celltypes = np.load('/data/nchand/analysis/BPcm/BP17_setup_data/ImmGenATAC1219.peak_matched_in_sorted.sl10004sh-4.celltypes.npy')
    lambda0, lambdap7 = get_data("/data/nchand/analysis/BPcm/BP17_L0_0/09-22-2023.08.38/analysis.npz", "/data/nchand/analysis/BPcm/BP17_L-1_7/05-26-2023.14.26/analysis.npz", 'bp_corr')
    print(lambda0.shape)
    scatter_plot_by_celltype(celltypes, lambda0, lambdap7, "$\lambda$=0", "$\lambda$=0.7", "/data/nchand/analysis/BPcm/BP17lambda0vs0.7_bp_correlation_1")
    
    # select_cells = ['B.FrE.BM']
    # mask = np.isin(celltypes, select_cells)
    # cell_idx = np.where(mask)[0]
    # eps = 1e-8

    # scatter_heat(lambda0[:,cell_idx][0]+eps, lambdap7[:, cell_idx][0]+eps, "$\lambda$=0", "$\lambda$=0.7", "/data/nchand/analysis/BPcm/BP17lambda0vs0.7_bp_correlation_B.FrE.BM_1")

if __name__ == '__main__':  
    plot_scalar_correlation()
    # plot_bp_correlation()
    




    # convert_to_txt("/data/nchand/analysis/BPcm/BP17_L0_0/09-22-2023.08.38/analysis.npz",
    #                'scalar_corr', 
    #                "/data/nchand/analysis/BPcm/BP17_L0_0/09-22-2023.08.38/scalar_correlations_for_plotting.txt")
    # convert_to_txt("/data/nchand/analysis/BPcm/BP17_L-1_7/05-26-2023.14.26/analysis.npz",
    #             'scalar_corr', 
    #             "/data/nchand/analysis/BPcm/BP17_L-1_7/05-26-2023.14.26/scalar_correlations_for_plotting.txt")