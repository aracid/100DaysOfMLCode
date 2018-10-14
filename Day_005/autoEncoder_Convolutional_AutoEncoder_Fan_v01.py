#https://blog.keras.io/building-autoencoders-in-keras.html

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np

import cv2
import glob


def loadImage():
    numberOfImages = len(glob.glob("C:/Users/brian/PycharmProjects/MachineLearning/Day_005/images/*.png"))
    trainTestRatio = int(numberOfImages*0.8)
    imageBuffer = []
    for image_path in glob.glob("C:/Users/brian/PycharmProjects/MachineLearning/Day_005/images/*.png"):
        image = cv2.imread(image_path, 0)
        # image = misc.imread(image_path, mode='L')
        imageBuffer.append(image)
    imageBuffer = np.array(imageBuffer)
    return imageBuffer[:trainTestRatio], imageBuffer[trainTestRatio:]


input_img = Input(shape=(64, 64,1), name='x_train') # adapt this if using `channels_first` image data format

x = Conv2D(16, (3,3), activation='relu', padding='same', name='first_encoded_conv')(input_img)
x = MaxPooling2D((2,2), padding='same',name='first_maxpool_conv')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same', name='second_encoded_conv')(x)
x = MaxPooling2D((2,2), padding='same', name='first_maxpooling_conv')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same', name='third_encoded_conv')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoded_third_maxpooling_conv')(x)


x = Conv2D(8,(3,3), activation='relu', padding='same', name='first_decoded_conv')(encoded)
x = UpSampling2D((2,2), name='first_upSampling_conv')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same', name='second_decoded_conv')(x)
x = UpSampling2D((2,2), name='second_upsampling_conv')(x)

x = Conv2D(16, (3,3), activation='relu', name='third_decoded_conv')(x)
x = UpSampling2D((2,2), name='third_upsampling_conv')(x)

decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same',name='Decoded_Conv_Out')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

x_train, x_test = loadImage()
print (x_train.shape)
print (x_test.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))  # adapt this if using `channels_first` image data format

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='C:/Users/brian/PycharmProjects/MachineLearning/Day_005/char-rnn-tensorflow-master/logs/autoencoder')])
decoded_imgs = autoencoder.predict(x_test)
#
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1,n,1):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(64, 64))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(64, 64))
#     # plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()