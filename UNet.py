import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
in_channels = 3
out_channels = 9
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.conv1 = Conv2d(n_channels, 64, kernel_size=3, padding=1) 
        self.conv11 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.down = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv6 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv9 = Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.up = nn.Upsample(scale_factor = (2, 2))
        self.conv10 = Conv2d(64, n_classes, kernel_size = 1)
        #self.merge = torch.cat((leftsidetensor, rightsidetensor), axis = 3)
    def forward(self, x):
        layer1 = relu(self.conv1(x))
        layer1 = relu(self.conv1(layer1))

        layer2down1 = self.down(layer1)
        layer2 = relu(self.conv2(layer2down1))
        layer2 = relu(self.conv2(layer2))

        layer3down2 = self.down(layer2)
        layer3 = relu(self.conv3(layer3down2))
        layer3 = relu(self.conv3(layer3))

        layer4down3 = self.down(layer3)
        layer4 = relu(self.conv4(layer4down3))
        layer4 = relu(self.conv4(layer4))

        layer5down4 = self.down(layer4)
        layer5 = relu(self.conv5(layer5down4))
        layer5 = relu(self.conv5(layer5))
        
        layer6up4 = self.up(layer5)
        skip4 = torch.cat((layer4, layer6up4))
        layer6up4 = relu(self.conv6(skip4))
        layer6up4 = relu(self.conv6(layer6up4))

        layer7up3 = self.up(layer6up4)
        skip3 = torch.cat((layer3, layer7up3))
        layer7up3 = relu(self.conv6(skip3))
        layer7up3 = relu(self.conv6(layer7up3))

        layer8up2 = self.up(layer7up3)
        skip2 = torch.cat((layer2, layer8up2))
        layer8up2 = relu(self.conv6(skip2))
        layer8up2 = relu(self.conv6(layer8up2))

        layer8up2 = self.up(layer7up3)
        skip2 = torch.cat((layer2, layer8up2))
        layer8up2 = relu(self.conv6(skip2))
        layer8up2 = self.conv6(layer8up2)

        layer9up1 = self.up(layer8up2)
        skip1 = self.merge4((layer1, layer9up1))
        layer9up1 = relu(self.conv6(skip1))
        layer9up1 = relu(self.conv6(layer9up1))
        return self.conv10(layer9up1)
#Create Network
net = UNet(n_channels = 3,n_classes = 9)
