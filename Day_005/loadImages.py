import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import misc
import glob
import numpy as np
imageBuffer = []

for image_path in glob.glob("C:/Users/brian/PycharmProjects/MachineLearning/Day_005/images/*.png"):
    image = misc.imread(image_path, mode='RGBA')
    # singleChannel = image[:,:,2]/255.0
    imageBuffer.append(image)


imageBuffer = np.array(imageBuffer)


n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n,1):
    # display original
    ax = plt.subplot(1, n, i)
    plt.imshow(imageBuffer[i])
    #plt.imshow(imageBuffer[i].reshape(28, 28))
   # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()