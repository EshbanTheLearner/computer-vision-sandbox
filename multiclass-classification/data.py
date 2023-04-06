from torchvision import datasets
from torchvision import transforms
from torchvision import utils
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import os

def download_data(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train = datasets.STL10(
        path, split="train",
        download=True, transform=transform
    )

    test = datasets.STL10(
        path, split="test",
        download=True, transform=transform
    )

    return train, test

def validation_test_split(test_set):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    indices = list(range(len(test_set)))
    y_test = [y for _, y in test_set]

    for test_index, val_index in sss.split(indices, y_test):
        print(f"Test: {test_index}, Val: {val_index}")
        print(f"Validation Set Size: {len(val_index)}, Test Set Size: {len(test_index)}")
    
    val = Subset(test_set, val_index)
    test = Subset(test_set, test_index)

    return val, test

def show(img, y=None, color=True):
    np_img = img.numpy()
    np_img_tr = np.transpose(np_img, (1, 2, 0))
    plt.imshow(np_img_tr)
    if y is not None:
        plt.title(f"Label: {str(y)}")

def plot_data(data):
    np.random.seed(42)
    grid_size = 4
    rnd_ids = np.random.randint(0, len(data), grid_size)
    x_grid = [data[i][0] for i in rnd_ids]
    y_grid = [data[i][1] for i in rnd_ids]
    x_grid = utils.make_grid(x_grid, nrows=4, padding=2)
    plt.figure(figsize=(10, 10))
    show(x_grid, y_grid)

def get_mean_RGB(data):
    meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in data]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return meanR, meanG, meanB

def get_std_RGB(data):
    stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in data]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return stdR, stdG, stdB
