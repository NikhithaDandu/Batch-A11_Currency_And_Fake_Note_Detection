

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# dialte image
def dilate_img(image):
    img = cv2.dilate(image, np.ones((5 ,5), np.uint8))
    return img


# convert image to grayscale
def img_to_gray(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return img_gray



# convert image to negative
def img_to_neg(image):
    img_neg = 255 - image
    return img_neg




# sobel edge detection
def sobel_edge2(image):
    # ksize = size of extended sobel kernel
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3, borderType = cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return dst


# binary thresholding
def binary_thresh(image, threshold):
    retval, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return img_thresh

