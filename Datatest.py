import numpy as np
from PIL import Image
data_dir = "./carseg_data/save"
DataTemp = np.load(data_dir+'/' + '0.npy')
img = Image.fromarray(DataTemp, 'RGB')
img.show()
