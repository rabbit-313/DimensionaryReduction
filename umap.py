import torch
import torchvision.models as models
import torchvision.transforms as transforms
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import umap
import warnings
from numba import NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)


def feature_plot(model, loader, config, path="feature_plot.png"):
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(config.device)
    labels = []
    features = []
    batch_size = config.batch_size

    with torch.no_grad():
        for _, _, batch in loader.iterator(batch_size=batch_size, shuffle=False, drop_last=False):
            x, y = batch._x, batch._y
            feature = feature_extractor(x)
            labels.append(y)
            features.append(feature.view(feature.size(0), -1))
    
    # pca = PCA(n_components=2)
    reducer = umap.UMAP()
    features_2d = reducer.fit_transform(torch.cat(features).cpu().numpy())
    labels = torch.cat(labels).cpu().numpy()
    print(f"labels:{labels}, label.shape:{labels.shape}")

    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1], color=colors[i], label=str(label), s=5)

    plt.legend()
    plt.savefig(path)
