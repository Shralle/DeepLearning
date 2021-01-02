import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from torchvision import transforms

data_dir_test = "./carseg_data/opel_Astra_no_segments_Camera_a-0001.png"
img = Image.open(data_dir_test)
new_image = img.resize((255, 255))
pix_val = list(new_image.getdata())
new_image.show()
pil_to_tensor = transforms.ToTensor()(new_image).unsqueeze_(0)
print(pil_to_tensor.shape) 

tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)

tensor_to_pil.show()




pix = img.load()
print(img.size)  # Get the width and hight of the image for iterating over
print(pix[300,400])  # Get the RGBA Value of the a pixel of an image
#pix[x,y] = value  # Set the RGBA Value of the image (tuple)
#img.save('alive_parrot.png')
image = imread(data_dir_test)
image_size = (255, 255)
image = resize(image, output_shape=image_size, mode='reflect', anti_aliasing=True)
plt.imshow(image)
plt.show()
pictureRGB = np.zeros((788,576,3))


