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