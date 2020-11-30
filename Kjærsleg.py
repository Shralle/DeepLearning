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
#import torch.from_numpy as from_numpy
import torch.optim as optim
#Initialize ARRAYS
inputs = np.ndarray(shape=(18,3,256,256), dtype=float)
labels = np.ndarray(shape=(18,10,256,256), dtype=float)
#Set directory for data
#data_dir = 'carseg_data/save'
data_dir = '/Users/frederikkjaer/Documents/DTU/DeepLearning/Projekt/DeepLearning/carseg_data/save'
#Initialize counter
i=0
#Loop for loading data
for filename in os.listdir(data_dir):
    datafiles = os.listdir(data_dir)
    if filename.endswith('.npy'):
        traindata = np.load(data_dir+'/' + filename)
        inputs[i] = traindata[0:3]
        labels[i] = traindata[3:13]
    i = i + 1       
#Convert data from nNumpy arrays into tensors 
inputs = torch.from_numpy(inputs).float()
labels = torch.from_numpy(labels).float()
#Initialize convolution layer size
in_channels = 3
mid_channels = 256
out_channels = 9
#Initialize Network
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
#Create Network
net = UNet(n_channels = 256,n_classes = 9)
#Ved ikke helt hvad det her gør, har bare brugt det før i tidligere ogpgaver :/
trainloader = torch.utils.data.DataLoader(inputs, batch_size=1, shuffle=True, num_workers=0)
batch = next(iter(trainloader))
labelloader = torch.utils.data.DataLoader(labels, batch_size=1, shuffle=True, num_workers=0)
#Optimizer / loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
#Training
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
        targets = labelloader
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
