from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, gray2rgb
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50


IMAGE_SIZE = [224, 224]

path = 'images/colorization/'

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in resnet.layers:
  layer.trainable=False   #We don't want to train these layers again, so False. 
resnet.summary()


#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)

train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=100000, class_mode=None)

 
#Convert from RGB to Lab

X =[]
Y =[]
for img in train[0]:
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0]) 
      Y.append(lab[:,:,1:] / 128) #A and B values range from -127 to 128, 
      #so we divide the values by 128 to restrict values to between -1 and 1.
  except:
     print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)


#now we have one channel of L in each layer but, ResNet50 is expecting 3 dimension, 
#so we repeated the L channel two times to get 3 dimensions of the same L channel

resnetfeatures = []
for i, sample in enumerate(X):
  sample = gray2rgb(sample)
  sample = sample.reshape((1,224,224,3))
  prediction = resnet.predict(sample)
  prediction = prediction.reshape((7, 7, 2048))
  resnetfeatures.append(prediction)
resnetfeatures = np.array(resnetfeatures)
print(resnetfeatures.shape)


#Decoder
model = Sequential()

model.add(Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(7,7,2048)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.summary()


model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
history = model.fit(resnetfeatures, Y, verbose=1, epochs=200, batch_size=100, validation_split=0.1)

model.save('colorize_autoencoder_ResNet50.model')
resnet.save('myresnet50.model')

accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy Model')
plt.legend()
plt.savefig('Accuracy.tiff', format='tiff', dpi=300)
plt.show()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Model')
plt.legend()
plt.savefig('Loss.tiff', format='tiff', dpi=300)
plt.show()