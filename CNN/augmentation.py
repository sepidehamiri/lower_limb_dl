# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:02:49 2020

@author: Sepideh Amiri
"""

import cv2
import glob
from matplotlib import pyplot
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# load the image
images_path = 'C:/Users/S&M/Desktop/cropped_normal/'
images = glob.glob(images_path + "*.png")
for img in images:
    img = load_img(img)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(horizontal_flip=True)
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
    # plot raw pixel data
    # pyplot.imshow(image)
    # show the figure
    # pyplot.show()
    cv2.imshow('ab', image)
    cv2.waitKey(0)
