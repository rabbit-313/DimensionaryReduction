import torch
import utils
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


label_name_dict = {'0': "brightness", '1': "contrast", '2': "defocus_blur", '3': "elastic_transform", '4': "fog",
                   '5': "frost", '6': "gaussian_noise", '7': "gaussian_blur", '8': "impulse_noise",
                   '9': "jpeg_compression", '10': "motion_blur", '11': "pixelate", '12': "shot_noise", '13': "snow", '14': "zoom_blur"}

label_color_dict = {'0': "red", '1': "blue", '2': "green", '3': "yellow", '4': "purple",
                    '5': "orange", '6': "pink", '7': "brown", '8': "gray",
                    '9': "olive", '10': "cyan", '11': "magenta", '12': "lime", '13': "teal", '14': "navy"}

def dim_reduction(latent_vecs, dim_labels, args):
    latent_vecs = latent_vecs.to("cpu")
    dim_labels = dim_labels.to("cpu")
    latent_vecs = latent_vecs.numpy()
    dim_labels = dim_labels.numpy()
    
    print("RedType", args.red_type)
    
    if args.red_type == "pca":
        latent_vecs_reduced = PCA(n_components=2, random_state=0).fit_transform(latent_vecs)
    elif args.red_type == "tsne":
        latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    elif args.red_type == "umap":
        latent_vecs_reduced = UMAP(n_components=2, random_state=0).fit_transform(latent_vecs)

    for i in range(len(label_name_dict)):
        plt.scatter(latent_vecs_reduced[dim_labels == i, 0], latent_vecs_reduced[dim_labels == i, 1], label=label_name_dict[str(i)], color=label_color_dict[str(i)], s=5)
        
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1),ncol=5, prop={'size': 7})
    
    if args.feature:
        plt.savefig(f'Results/BeforeLiner/{args.data_name}-{args.red_type}.png') 
    else:
        plt.savefig(f'Results/AfterLiner/{args.data_name}-{args.red_type}.png')
    plt.show()

def main(args):
    testdata_path = args.test_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_dict = {
        'test': transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #  transforms.CenterCrop(size=(64, 64)),
             ])
    }
    
    # データ準備
    test_dataset = ImageFolder(root=testdata_path, transform=transform_dict['test'])
    test_loader_eval = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # model
    net = utils.get_resnet_model(resnet_type=152, pretrained=False)
    net.fc = torch.nn.Linear(net.fc.in_features, 15)
    model = net.to(device)
    model.load_state_dict(torch.load(args.model))

    # モデルの評価
    if args.eval:
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for imgs, labels in tqdm(test_loader_eval, desc=f"Test", unit="batch"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                _, outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
    
    if args.data_name == "Tiny-ImageNet-C5":
        model.eval()
        with torch.no_grad():
            feature = []
            dim_labels = []
            for imgs, labels in tqdm(test_loader_eval, desc=f"DimReduction", unit="batch"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                if args.feature:
                    outputs, _ = model(imgs)
                else:
                    _, outputs = model(imgs)
                feature.append(outputs)
                dim_labels.append(labels)
            feature = torch.cat(feature, dim=0)
            dim_labels = torch.cat(dim_labels, dim=0)
            dim_reduction(feature, dim_labels, args)
    else:
        model.eval()
        with torch.no_grad():
            feature = []
            predicted_labels = []
            for imgs, _ in tqdm(test_loader_eval, desc=f"DimReduction", unit="batch"):
                imgs = imgs.to(device)
                _, outputs = model(imgs)
                if args.feature:
                    feature_outputs, _ = model(imgs)
                    feature.append(feature_outputs)
                else:
                    feature.append(outputs)
                    
                if args.batch_size != 1:
                    _, predicted = torch.max(outputs, dim=1)
                else:
                    _, predicted = torch.max(outputs, dim=0)
                predicted_labels.append(predicted)
                
            if args.batch_size != 1:
                feature = torch.cat(feature, dim=0)
                predicted_labels = torch.cat(predicted_labels, dim=0)
            else:
                feature = torch.stack(feature)
                predicted_labels = torch.cat([tensor.view(-1) for tensor in predicted_labels])
            dim_reduction(feature, predicted_labels, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dataset', type=str)  
    parser.add_argument('--model', type=str, default='/home/ru/ドキュメント/study/DegradationClassification/results/ex4/ex4_17.pt')   
    parser.add_argument('--red_type', type=str, default="tsne")
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--feature', type=bool, default=False)
    args = parser.parse_args()
    
    print(args)
    
    pl.seed_everything(0)
    main(args)
