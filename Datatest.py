import numpy as np
from PIL import Image
data_dir = "./carseg_data/save"
DataTemp = np.load(data_dir+'/' + '0.npy')
test = np.arange(0,257)
meantest = np.mean(test)
stdtest = np.std(test)
meanRPG = [meantest, meantest, meantest]
std = [stdtest, stdtest, stdtest]
PicturePrint = np.zeros((256,256,3), dtype = float)
for i in range(0,256):
    for j in range(0,256):
        DataTemp[0:3,i,j] = (DataTemp[0:3,i,j]*std)+meanRPG
        PicturePrint[i,j,:] = DataTemp[0:3,i,j]
img = Image.fromarray(PicturePrint, 'RGB')
img.show()
 

 

 

