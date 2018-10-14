#https://blog.keras.io/building-autoencoders-in-keras.html

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from keras import regularizers
from keras.callbacks import TensorBoard
import time

EPOCHS = 100
BATCH_SIZE = 1024
FUTURE_PERIOD_PREDICT = "OBJECT"
NAME = f"{EPOCHS}-SEQ-{FUTURE_PERIOD_PREDICT}-AutoEncoder-{int(time.time())}"


import cv2
import glob


def loadImage():
    numberOfImages = len(glob.glob("C:/Users/brian/PycharmProjects/MachineLearning/Day_005/images/small/*.png"))
    trainTestRatio = int(numberOfImages*0.8)
    imageBuffer = []
    for image_path in glob.glob("C:/Users/brian/PycharmProjects/MachineLearning/Day_005/images/small/*.png"):
        image = cv2.imread(image_path, 0)
        # image = misc.imread(image_path, mode='L')
        imageBuffer.append(image)
    imageBuffer = np.array(imageBuffer)
    return imageBuffer[:trainTestRatio], imageBuffer[trainTestRatio:]


# This is the size of our encoded representations
encoding_dim = 64 # 32 Floats -> Compression factor of 24.5 assuming the input is 784 floats

# This is our input placeholder
input_img = Input(shape=(784,))

# "Encoded" is the lossy reconstruction of the input

encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
#encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-6))(input_img)

#encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='sigmoid')(encoded)
decoded = Dense(128, activation='sigmoid')(encoded)
decoded = Dense(256, activation='sigmoid')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

#retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
opt = Adam(lr=0.001, decay=1e-6)
autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

tensorboard = TensorBoard(log_dir=f'C:/Users/brian/PycharmProjects/MachineLearning/Day_005/logs/{NAME}')

x_train, x_test = loadImage()
print (x_train.shape)
print (x_test.shape)

# (x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# print(len(x_train ))
# print(len(x_test ))


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=1028,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard])

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10

plt.figure(figsize=(20,4))

for i in range(n):
    # display original
    ax = plt.subplot(2, n , i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()