import os
import torch
import numpy as np
import torch.nn as nn
from UNetSimple import UNet
import torch.optim as optim
from SoftDiceloss import SoftDiceloss
from dice_loss import dice_loss
#from ConvolutionNetwork import Convolution
from torch.utils.data import DataLoader, random_split
#import matplotlib.pyplot as plt
#from PIL import Image


#Set directory for data
data_dir = "./carseg_data/savebig"
#Initialize ARRAYSdataset_size = len(os.listdir(data_dir))
dataset_size = len(os.listdir(data_dir))
DataAll= np.ndarray(shape=(dataset_size,13,256,256), dtype = float)
#Initialize counter
i=0
#Loop for loading data
for filename in os.listdir(data_dir):
    if filename.endswith('.npy'):
        DataTemp = np.load(data_dir+'/' + filename)
        DataAll[i] = DataTemp

    i = i + 1       
torch.manual_seed(0)
data = torch.from_numpy(DataAll).float()
n_test = int(len(data) * 0.1)
n_train = len(data) - n_test
train, test = random_split(data, [n_train, n_test], generator = torch.manual_seed(0))

batch_size = 5

#Splits the data intop batches
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False)

in_channels = 3
mid_channels = 256
#Initialize Network

#Create Network
net = UNet(n_channels = 3,n_classes = 9)

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

if use_cuda:
    net.cuda()
print(net)

#Optimizer / loss function
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
#Training
from torch.autograd import Variable
num_epoch = 40
for epoch in range(num_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        # wrap them in Variable
        # zero the parameter gradients
        # Your code here!
        inputs = data[:,0:3,:,:]
        labels = data[:,3:12,:,:]
        #mask = data[:,12,:,:]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        labels = torch.argmax(labels, dim = 1)
        outputs = net(inputs)
        #loss = criterion(outputs, targets)
        loss = dice_loss(labels, outputs, ignore_background=True)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
print('Finished Training')

PATH = './model/model_40epoch5batchV2.pt'

torch.save(net, PATH)