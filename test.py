import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import os

model = tf.keras.models.load_model('encoder-decoder/results/final_colorize_autoencoder.model',
                                   custom_objects=None,
                                   compile=True)

testpath = 'images/colorization/test/gray/'
files = os.listdir(testpath)


for idx, file in enumerate(files):
    img1_color=[]
    img1=img_to_array(load_img(testpath+file))
    img1 = resize(img1 ,(256,256))
    img1_color.append(img1)
    
    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))
    
    output1 = model.predict(img1_color)
    output1 = output1*128
    
    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]
    imsave('images/colorization/test/result/result'+str(idx+1)+".jpg", lab2rgb(result))