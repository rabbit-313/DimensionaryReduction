from __future__ import print_function
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import utils


def test(t_SNE=True):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    testdata_path = os.path.join(base_path, "Tiny-ImageNet-C5-ImageFolder", "Test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_dict = {
        'test': transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ])
    }
    
    test_dataset = ImageFolder(root=testdata_path, transform=transform_dict['test'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    imgs = []
    labels = []
    count = 0
    for img, label in test_loader:
        imgs.append(img)
        labels.append(label)
        count+=1
        if count==5000:
            break    

    # model
    net = utils.get_resnet_model(resnet_type=152, pretrained=False)
    model = net.to(device)
    model.load_state_dict(torch.load('/home/ru/ドキュメント/study/DegradationClassification/results/ex4/ex4_17.pt'))
    classes = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_categories = [] #予想ラベルたち
        target_array = label.numpy()
        master_features = [] #マスター画像の埋め込みたち
        for i in classes:
            indexes = np.where(target_array==int(i))[0]
            master_img = data[np.random.choice(indexes)].to(device)
            master_img = torch.unsqueeze(master_img, dim=0)
            master_img = master_img.to(device)
            embedded_master_img = model(master_img)
            master_features.append(embedded_master_img)
        master_features = torch.cat(master_features) # (10, 128)

        data = data.to(device)
        output = model(data)
        output_unbind = torch.unbind(output)

    for embedded_img in output_unbind:
        distances = torch.sum((master_features - embedded_img)**2, dim=1) #(10)
        pred_category = classes[distances.argmin()]
        pred_categories.append(int(pred_category))
        
    pred_category = torch.LongTensor(pred_categories)
    # ラベルが数字だったのでtorch.Tensorにして条件文でやった。strならforぶん回す
    correct += (labels == pred_category).sum()
    accuracy = float(correct)*100 / len(pred_categories)

    print('Accuracy: {}/{} ({}%)\n'.format(correct, len(pred_categories), accuracy))

    if t_SNE:
        t_sne(output, labels)


def t_sne(latent_vecs, target):
    latent_vecs = latent_vecs.to("cpu")
    latent_vecs = latent_vecs.numpy()
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    # latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1], c=target, cmap='jet')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    test()
