from load_model import load_model
from eval_model import get_model_structure
from tangermeme.utils import random_one_hot
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.ersatz import substitute
import torch
from AttributionModelWrapper import AttributionModelWrapper
from matplotlib import pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
from tangermeme.plot import plot_logo
import inspect


# wrap the model in a another wrapper model that outputs predictions of size (batch_size, n_targets)

def get_logo(X_attr):
    fig = plt.figure(figsize=(10, 2))
    ax = plt.subplot(111)

    plot_logo(X_attr[0, :, 480:625], ax=ax)
    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig('test_logo', dpi=300, bbox_inches='tight')



def main():
    seq_len=998
    X = random_one_hot((1, 4, seq_len)).type(torch.float32)
    X = substitute(X, "GTGACTCATC")
    print(X[:, :, 490: 520])
    print(X.size())

    # figure out a way to embed 
    

    saved_model_path = "/data/nchand/analysis/BPcm/BP68_L-1_5/complete/05-09-2024.12.10/best_model"
    structure = get_model_structure('BPcm', 300, 90)
    model = load_model(saved_model_path, model_structure=structure, n_filters=300)
    wrapped_model = AttributionModelWrapper(model)


    X_attr = deep_lift_shap(wrapped_model, X, target=0, device='cpu', random_state=0)
    X_attr.shape
    print(X_attr.shape)

    non_zero_count = torch.count_nonzero(X_attr)
    print(f"Number of non-zero values in X_attr: {non_zero_count.item()}")

    get_logo(X_attr)

if __name__ == "__main__":
    main()
    





