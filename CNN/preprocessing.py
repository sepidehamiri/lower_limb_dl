# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:36:16 2020

@author: SM
"""

import os
import cv2
import glob
import Image
import numpy as np

# loading train images
images_path = 'C:/Users/S&M/Desktop/'
images = glob.glob(images_path + "*.png")

n = 3
i = 10027
max_h = 0
max_w = 0


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


for img in images:
    image = cv2.imread(img)
    rotate_image = rotate_image(image, 15)
    h, w, _ = image.shape
    if h > max_h:
        max_h = h

    if w > max_w:
        max_w = w
    new_width_start = (h // (n + 3)) + 12
    new_width_end = ((n - 1) * h // n) + 23
    crop_img = image[0:h, new_width_start:new_width_end]
    path = 'C:/Users/S&M/Desktop/nor/'
    imageName = str(i) + ".png"
    cv2.imwrite(os.path.join(path, imageName), image)
    i = i + 1
    cv2.imshow('ab', rotate_image)
    cv2.waitKey(0)

print(max_h, max_w)

path = 'C:/Users/S&M/Desktop/0.png'
image = cv2.imread(path)
h, w, _ = image.shape
print(h, w)
old_im = Image.open(path)
old_size = old_im.size

new_size = (800, 800)
new_im = Image.new("RGB", new_size)
new_im.paste(old_im, ((new_size[0] - old_size[0]) / 2, (new_size[1] - old_size[1]) / 2))

new_im.show()
new_im.save('someimage.jpg')
