<<<<<<< HEAD
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import models
from torchvision import transforms

from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import collections
import copy

import data as d

path2data = "./data"

train_ds, test_ds = d.download_data(path2data)

print(train_ds.data.shape)
print(test_ds.data.shape)

val_ds, test_ds = d.validation_test_split(test_ds)

y_train = [y for _, y in train_ds]
y_val = [y for _, y in val_ds]
y_test = [y for _, y in test_ds]

counter_train = collections.Counter(y_train)
counter_val = collections.Counter(y_val)
counter_test = collections.Counter(y_test)

print(counter_train)
print(counter_val)
print(counter_test)

meanR, meanG, meanB = d.get_mean_RGB(train_ds)
stdR, stdG, stdB = d.get_std_RGB(train_ds)

train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [meanR, meanG, meanB],
        [stdR, stdG, stdB]
    )
])

test_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [meanR, meanG, meanB],
        [stdR, stdG, stdB]
    )
])

train_ds.transforms = train_transformer
=======
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import models

from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import collections
import copy

import data as d

path2data = "./data"

train_ds, test_ds = d.download_data(path2data)

print(train_ds.data.shape)
print(test_ds.data.shape)

val_ds, test_ds = d.validation_test_split(test_ds)

y_train = [y for _, y in train_ds]
y_val = [y for _, y in val_ds]
y_test = [y for _, y in test_ds]

counter_train = collections.Counter(y_train)
counter_val = collections.Counter(y_val)
counter_test = collections.Counter(y_test)

print(counter_train)
print(counter_val)
print(counter_test)
>>>>>>> 179f0cf1dfedd64b2b8a906a65f68000602bbbe5
