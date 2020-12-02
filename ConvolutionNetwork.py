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

#Set directory for data
data_dir = '/Users/frederikkjaer/Documents/DTU/DeepLearning/Projekt/DeepLearning/carseg_data/save'
#mus dir:
#data_dir = "/Users/Rnd/Documents/DeepLearning/DeepLearning/carseg_data/save"
#Initialize ARRAYS

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
batch_size = 6

#Splits the data intop batches
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False)

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
#Create Network
net = Convolution(n_channels = 3,n_classes = 9)

#Optimizer / loss function
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
#Training
from torch.autograd import Variable
num_epoch = 1
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
        mask = data[:,12,:,:]
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
    inputs = data[:,0:3,:,:]
    labels = data[:,3:12,:,:]
    mask = data[:,12,:,:]
    labels = torch.argmax(labels, dim = 1)
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #correct += (predicted == labels).sum()
    correct += torch.sum(predicted == labels)
#print('Accuracy of the network on the {} test images: {:4.2f} %'.format(n_test, (100 * correct.true_divide(total*256*256))))
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
pictureprint = np.zeros((3,256,256))
for i in range(256):
    for j in range(256):
        if(picture[i,j] == 0):
            pictureprint[:,i,j] = colors[0]
        if(picture[i,j] == 1):
            pictureprint[:,i,j] = colors[1]
        if(picture[i,j] == 2):
            pictureprint[:,i,j] = colors[2]
        if(picture[i,j] == 3):
            pictureprint[:,i,j] = colors[3]
        if(picture[i,j] == 4):
            pictureprint[:,i,j] = colors[4]
        if(picture[i,j] == 5):
            pictureprint[:,i,j] = colors[5]
        if(picture[i,j] == 6):
            pictureprint[:,i,j] = colors[6]
        if(picture[i,j] == 7):
            pictureprint[:,i,j] = colors[7]
        if(picture[i,j] == 8):
            pictureprint[:,i,j] = colors[8]
        if(picture[i,j] == 9):
            pictureprint[:,i,j] = colors[9]
plt.imshow(picture)