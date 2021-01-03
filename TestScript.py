import torch.nn as nn
#from PIL import Image
#from ConvolutionNetwork import Convolution
from UNetSimple import UNet
from torch.autograd import Variable
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

#Set directory for data
#HPC
data_dir = "./carseg_data/save"

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
batch_size = 1
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False)


PATH = './model/model_40epoch5batchV3.pt'
net = UNet(n_channels = 3,n_classes = 9)

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")
if use_cuda:
    net.cuda()
#print(net)

net = torch.load(PATH)

#model.eval()

correct = 0
total = 0
predictedlist = torch.zeros([len(test_loader), 256, 256], dtype=torch.float64)
labellist = torch.zeros([len(test_loader), 256, 256], dtype=torch.float64)
count = 0
nobackground = 0

for data in test_loader:
    inputs = data[:,0:3,:,:]
    labels = data[:,3:12,:,:]
    mask = data[:,12,:,:]
    ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs, labels, mask = inputs.to(device), labels.to(device) , mask.to(device)
    labels = torch.argmax(labels, dim = 1)
    ###
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #correct += (predicted == labels).sum()
    nobackground += torch.sum(labels == 0)
    correct += torch.sum(predicted == labels)
    predictedlist[count, :, :] = predicted
    labellist[count, :, :] = labels
    count = count + 1

#print('Accuracy of the network on the {} test images: {:4.2f} %'.format(n_test, (100 * correct.true_divide(total*256*256))))
print(nobackground.item())
print('Accuracy of the network on the {} test images: {:4.2f} %'.format(n_test, (100 * correct.true_divide(total*256*256-nobackground.item()))))

colors = {0: [int(0), int(0), int(0)],
          1: [int(10), int(100), int(10)],
          2: [int(250), int(250), int(10)],
          3: [int(10), int(10), int(250)],
          4: [int(10), int(250), int(250)],
          5: [int(250), int(10), int(250)],
          6: [int(250), int(150), int(10)],
          7: [int(150), int(10), int(150)],
          8: [int(10), int(250), int(10)]}
print(predictedlist.shape)
#picture = predictedlist[1,:,:]
picture = labellist[1,:,:]
pictureprint = np.zeros((256,256,3),dtype=int)
for i in range(256):
    for j in range(256):
        if(picture[i,j] == 0):
            pictureprint[i,j,:] = (colors[0])
        if(picture[i,j] == 1):
            pictureprint[i,j,:] = (colors[1])
        if(picture[i,j] == 2):
            pictureprint[i,j,:] = (colors[2])
        if(picture[i,j] == 3):
            pictureprint[i,j,:] = (colors[3])
        if(picture[i,j] == 4):
            pictureprint[i,j,:] = (colors[4])
        if(picture[i,j] == 5):
            pictureprint[i,j,:] = (colors[5])
        if(picture[i,j] == 6):
            pictureprint[i,j,:] = (colors[6])
        if(picture[i,j] == 7):
            pictureprint[i,j,:] = (colors[7])
        if(picture[i,j] == 8):
            pictureprint[i,j,:] = (colors[8])
        if(picture[i,j] == 9):
            pictureprint[i,j,:] = (colors[9])
plt.imshow(pictureprint, interpolation='nearest')
plt.show()