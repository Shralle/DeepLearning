#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:17:29 2020

@author: frederikkjaer
"""
import os
#import glob
import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
import torch.from_numpy as from_numpy
import torch.optim as optim
traindata = []
inputs = np.ndarray(shape=(3,256,256), dtype=float)
labels = np.ndarray(shape=(10,256,256), dtype=float)
labels_list = []
inputs_list = []
labels = []
data_dir = 'carseg_data/save'
for filename in os.listdir(data_dir):
    datafiles = os.listdir(data_dir)
    if filename.endswith('.npy'):
        traindata = np.load(data_dir+'/' + filename)
        inputs = traindata[0:3]
        inputs_list.append(inputs)
        labels = traindata[3:13]
        labels_list.append(labels)
print(inputs_list[17][2])
in_channels = 256
mid_channels = 256
out_channels = 256
test = torch.from_numpy(inputs_list[0])
print(test)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.double_conv(x)
net = UNet(n_channels = 256,n_classes = 9)
inputs_list_tensor = []
for i in range(len(inputs_list)-1):
    inputs_list_tensor.append( torch.from_numpy(inputs_list[i]))
print(inputs_list_tensor[0])
trainloader = torch.utils.data.DataLoader(inputs_list_tensor, batch_size=1, shuffle=True, num_workers=2)
batch = next(iter(trainloader))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
num_epoch = 3
for epoch in range(num_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        # wrap them in Variable
        # zero the parameter gradients
        # Your code here!
        optimizer.zero_grad()
        targets = labels
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.forward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
print('Finished Training')
