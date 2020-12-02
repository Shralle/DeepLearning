import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.conv1 = Conv2d(n_channels, 64, kernel_size=3, padding=1) 
        self.conv11 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.down = MaxPool2d(kernel_size=2, stride = 2)
        self.conv2 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = Conv2d(128,128, kernel_size=3, padding=1)
        self.conv3 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv33 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv44 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv55 = Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv6up4 = Conv2d(1024,512, kernel_size = 3, padding = 1)
        self.conv6 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7up3 = Conv2d(512,256, kernel_size = 3, padding = 1)
        self.conv7 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8up2 = Conv2d(256,128, kernel_size = 3, padding = 1)
        self.conv8 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv9up1 = Conv2d(128,64, kernel_size = 3, padding = 1)
        self.conv9 = Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.up = nn.Upsample(scale_factor = 2)
        self.conv10 = Conv2d(64, n_classes, kernel_size = 1)
        #self.merge = torch.cat((leftsidetensor, rightsidetensor), axis = 3)
    def forward(self, x):
        layer1 = relu(self.conv1(x))
        layer1 = relu(self.conv11(layer1))

        layer2down1 = self.down(layer1)
        layer2 = relu(self.conv2(layer2down1))
        layer22 = relu(self.conv22(layer2))

        layer3down2 = self.down(layer22)
        layer3 = relu(self.conv3(layer3down2))
        layer33 = relu(self.conv33(layer3))

        layer4down3 = self.down(layer33)
        layer4 = relu(self.conv4(layer4down3))
        layer44 = relu(self.conv44(layer4))

        layer5down4 = self.down(layer44)
        layer5 = relu(self.conv5(layer5down4))
        layer55 = relu(self.conv55(layer5))
        
        layer6up4 = self.up(layer55)
        layer6up4 = self.conv6up4(layer6up4)
        skip4 = torch.cat((layer4, layer6up4))
        layer6up4 = relu(self.conv6(skip4))
        layer6up4 = relu(self.conv6(layer6up4))

        layer7up3 = self.up(layer6up4)
        layer7up3 = self.conv7up3(layer7up3)
        skip3 = torch.cat((layer3, layer7up3))
        layer7up3 = relu(self.conv7(skip3))
        layer7up3 = relu(self.conv7(layer7up3))

        layer8up2 = self.up(layer7up3)
        layer8up2 = self.conv8up2(layer8up2)
        skip2 = torch.cat((layer2, layer8up2))
        layer8up2 = relu(self.conv8(skip2))
        layer8up2 = relu(self.conv8(layer8up2))

        layer9up1 = self.up(layer8up2)
        layer9up2 = self.conv9up1(layer9up1)
        skip1 = torch.cat((layer1, layer9up2))
        layer9up1 = relu(self.conv9(skip1))
        layer9up1 = relu(self.conv9(layer9up1))
        return self.conv10(layer9up1)
