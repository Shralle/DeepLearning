import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import tensorflow 
in_channels = 3
mid_channels = 256
out_channels = 9
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
            self.conv1 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.down = nn.MaxPool2d(pool_size=(2, 2))
            self.conv2 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.conv3 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.conv4 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.conv5 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.conv6 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.conv7 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.conv8 = Conv2d(n_channels, mid_channels, activation = 'relu' ,kernel_size=3, padding=1)
            self.up = tf.keras.layers.UpSampling2D(size=(2, 2))
            self.Convup4 = Conv2D(512, kernal_size = 2, activation = 'relu', padding = 1, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))
    def forward(self, x):
        layer1 = self.conv1(x)
        layer1 = self.conv1(layer1)
        layer1down1 = self.down(layer1)
        layer2 = self.conv2(layer1down1)
        layer2 = self.conv2(layer2)
        layer2down2 = self.down(layer2)
        layer3 = self.conv3(layer2down2)
        layer3 = self.conv3(layer3)
        layer3down3 = self.down(layer3)
        layer4 = self.conv4(layer3down3)
        layer4 = self.conv4(layer4)
        layer4down4 = self.down(layer4)
        layer5 = self.conv5(layer4down4)
        layer5 = slef.conv5(layer5)
        layer6.up4
        return self.double_conv(x)
#Create Network
net = UNet(n_channels = 3,n_classes = 9)
