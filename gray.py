from skimage.color import rgb2lab
from skimage.io import imsave
from keras.preprocessing.image import img_to_array, load_img
import os

imagepath = 'C:/Users/Ripon/Desktop/Spider/images/colorization/test/or/ed/'
files = os.listdir(imagepath)
for index, file in enumerate(files):
    
    test = img_to_array(load_img(imagepath+file))
    
    lab = rgb2lab(test)    
    L = lab[:,:,0]
    A = lab[:,:,1]
    B = lab[:,:,2]
    
    imsave('C:/Users/Ripon/Desktop/Spider/images/colorization/test/gry/ed/gray'+str(index+1)+".jpg", L)