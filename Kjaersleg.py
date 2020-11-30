import os
#import glob
import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
#import torch.from_numpy as from_numpy
import torch.optim as optim
#Set directory for data
#data_dir = 'carseg_data/save'
data_dir_train = '/Users/frederikkjaer/Documents/DTU/DeepLearning/Projekt/DeepLearning/carseg_data/save'
data_dir_test = '/Users/frederikkjaer/Documents/DTU/DeepLearning/Projekt/DeepLearning/carseg_data/test'
#mus dir:
#data_dir = "/Users/Rnd/Documents/DeepLearning/DeepLearning/carseg_data/save"
#Initialize ARRAYS
train_dataset_size = len(os.listdir(data_dir_train))
test_dataset_size = len(os.listdir(data_dir_test))
train = np.ndarray(shape=(train_dataset_size,13,256,256), dtype = float)
test = np.ndarray(shape=(test_dataset_size,13,256,256), dtype = float)
#Initialize counter
i=0
j = 0
#Loop for loading data
for filename in os.listdir(data_dir_train):
    datafiles = os.listdir(data_dir_train)
    if filename.endswith('.npy'):
        traindata = np.load(data_dir_train+'/' + filename)
        train[i] = traindata

    i = i + 1       
#Convert data from nNumpy arrays into tensors 
for filename in os.listdir(data_dir_test):
    datafiles = os.listdir(data_dir_test)
    if filename.endswith('.npy'):
        testdata = np.load(data_dir_test+'/' + filename)
        test[j] = testdata

    j = j + 1   
train = torch.from_numpy(train).float()
test = torch.from_numpy(test).float()

in_channels = 3
mid_channels = 256
out_channels = 9
#Initialize Network
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
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
#Create Network
net = UNet(n_channels = 3,n_classes = 9)

#Seperates the data into batches of size 6.
trainloader = torch.utils.data.DataLoader(train, batch_size = 6, shuffle=True)
train_data_iter = next(iter(trainloader))
testloader = torch.utils.data.DataLoader(test, batch_size = 6, shuffle=True)

#Optimizer / loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
#Training
from torch.autograd import Variable
num_epoch = 3
for epoch in range(num_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        # wrap them in Variable
        # zero the parameter gradients
        # Your code here!
        inputs = data[:,0:3,:,:]
        labels = data[:,4:13,:,:]
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        targets = labels
        targets = torch.argmax(targets, dim = 1)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
print('Finished Training')

correct = 0
total = 0

for data in testloader:
    images = data[:,0:3,:,:]
    labels = data[:,4:13,:,:]
    labels = torch.argmax(labels, dim = 1)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the {} test images: {:4.2f} %'.format(test_dataset_size, (100 * correct.true_divide(total*256*256))))