#!/usr/bin/env python
# coding: utf-8

# In[3]:


from utils import *
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image, ImageTk



def impli(file_path):
    TRAIN_DIR = 'C:/Users/divya/Desktop/fake_note_project/data/training'
    test_img = cv2.imread(file_path)

    classes = 7
    labels = ['10','20','50','100','200','500','2000']
    training_set = []
    image_set = []
    for i in range(classes):
        path = os.path.join(TRAIN_DIR,labels[i])
        images = os.listdir(path)
        for j in images:
            img = path+"/"+j;
            img1 = cv2.imread(img)
            img1 = cv2.resize(img1, (100,100))
            image_set.append(img1)
            training_set.append(img)
            
            
    resized_img = cv2.resize(test_img,(100,100))                               #resizing the input image
    dilate = dilate_img(test_img)                                              #applying dilation
    grey_img = img_to_gray(dilate)                                             #grayscale conversion
    neg_img = img_to_neg(grey_img)                                             #negative image conversion
    edge_detect = sobel_edge2(neg_img)                                         #edge detection
    bin_thresh = binary_thresh(edge_detect,190)                                #thresholding
    
        
    orb = cv2.ORB_create()
    (kp1, des1) = orb.detectAndCompute(test_img, None)
    max_val=25

    for i in range(0, len(training_set)):
        # train image
        train_img = cv2.imread(training_set[i])

        (kp2, des2) = orb.detectAndCompute(train_img, None)

        # brute force matcher
        bf = cv2.BFMatcher()
        all_matches = bf.knnMatch(des1, des2, k=2)
        good = []

        # if good -> append to list of good matches
        for (m,n) in all_matches:
            if m.distance < 0.789 * n.distance:
                good.append([m])

        if len(good) > max_val:
            max_val = len(good)
            
        
    if max_val > 25:
        return 'Original Note'

    else:
        return 'Fake Note'
        


# In[ ]:




