import os
import torch
import numpy as np
import torch.nn as nn
from UNetSimple import UNet
import torch.optim as optim
from SoftDiceloss import SoftDiceloss
from dice_loss import dice_loss
from ConvolutionNetwork import Convolution
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

#Set directory for data
data_dir = '/Users/frederikkjaer/Documents/DTU/DeepLearning/Projekt/DeepLearning/carseg_data/save'
#data_dir_test = '/Users/frederikkjaer/Documents/DTU/DeepLearning/Projekt/DeepLearning/carseg_data/test'
#mus dir:
#data_dir = "/Users/Rnd/Documents/DeepLearning/DeepLearning/carseg_data/save"
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

data = torch.from_numpy(DataAll).float()
n_test = int(len(data) * 0.1)
n_train = len(data) - n_test
train, test = random_split(data, [n_train, n_test])
batch_size = 30

#Splits the data intop batches
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=True)

net = Convolution(n_channels = 3,n_classes = 9)
#Optimizer / loss function
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
#Training
from torch.autograd import Variable
num_epoch = 3
for epoch in range(num_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        # wrap them in Variable
        # zero the parameter gradients
        # Your code here!
        inputs = data[:,0:3,:,:]
        mask = data[:,3,:,:]
        labels = data[:,4:13,:,:]
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        targets = labels
        targets = torch.argmax(targets, dim = 1)
        outputs = net(inputs)
        #loss = criterion(outputs, targets)
        loss = dice_loss(targets, outputs, ignore_background=True)
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

for data in test_loader:
    images = data[:,0:3,:,:]
    labels = data[:,4:13,:,:]
    labels = torch.argmax(labels, dim = 1)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the {} test images: {:4.2f} %'.format(n_test, (100 * correct.true_divide(total*256*256))))
colors = {0: [0, 0, 0],
          1: [10, 100, 10],
          2: [250, 250, 10],
          3: [10, 10, 250],
          4: [10, 250, 250],
          5: [250, 10, 250],
          6: [250, 150, 10],
          7: [150, 10, 150],
          8: [10, 250, 10]}
print(colors[1])
picture = predicted[1]
for i in range(0,8):
    if(i == 0):
        picture[i,:,:] = picture[colors[0],:,:]
    if(i == 1):
        picture[i,:,:] = picture[colors[1],:,:]
    if(i == 2):
        picture[i,:,:] = picture[colors[2],:,:]
    if(i == 3):
        picture[i,:,:] = picture[colors[3],:,:]
    if(i == 4):
        picture[i,:,:] = picture[colors[4],:,:]
    if(i == 5):
        picture[i,:,:] = picture[colors[5],:,:]
    if(i == 6):
        picture[i,:,:] = picture[colors[6],:,:]
    if(i == 7):
        picture[i,:,:] = picture[colors[7],:,:]
    if(i == 8):
        picture[i,:,:] = picture[colors[8],:,:]
    if(i == 9):
        picture[i,:,:] = picture[colors[9],:,:]
plt.imshow(picture)