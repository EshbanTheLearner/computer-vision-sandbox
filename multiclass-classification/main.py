import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import datasets
from torchvision import utils
import torchvision.transforms as transforms
from torchvision import models

from torchsummary import summary

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import collections
import copy

import data as d

path2data = "./data"

train_ds, test_ds = d.download_data(path2data)
# print(test_ds.data.shape)

val_ds, test_ds = d.validation_test_split(test_ds)

print(train_ds.data.shape)
print(val_ds.data.shape)
print(test_ds.data.shape)


y_train = [y for _, y in train_ds]
y_val = [y for _, y in val_ds]
y_test = [y for _, y in test_ds]

counter_train = collections.Counter(y_train)
counter_val = collections.Counter(y_val)
counter_test = collections.Counter(y_test)

print(counter_train)
print(counter_val)
print(counter_test)