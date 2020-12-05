import os
#import glob
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from SoftDiceloss import SoftDiceloss
from dice_loss import dice_loss
import matplotlib.pyplot as plt
from PIL import Image

in_channels = 3
mid_channels = 256
#Initialize Network
class Convolution(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Convolution, self).__init__()
        self.double_conv = nn.Sequential(
                    nn.Conv2d(n_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, n_classes, kernel_size=3, padding=1),
                    nn.BatchNorm2d(n_classes),
                    nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.double_conv(x)
