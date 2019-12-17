# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 00:06:10 2019

@author: Siddharth
"""
#This file mainly has the pre-processing functions which is needed before applying the feature extraction.

import cv2
import scipy

class Extraction:
    def __init__(self):
        pass
    
    #coverting the image into YUV using the openCV library function cvtCOLOR
    @staticmethod
    def convert_to_yuv(img_path):
        image=cv2.imread(img_path)
        yuv_image=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        return yuv_image
    
    #spliting the image into 192 100x100 windows by using simple for loop and image shape
    @staticmethod
    def split_image_into_windows(img):
        img_windows= [img[x:x+100,y:y+100] for x in range(0, img.shape[0], 100) for y in range(0,img.shape[1], 100)]
        return img_windows
        
    #converting the image into Grayscale using the openCV library function cvtCOLOR
    @staticmethod
    def convert_to_grayscale(img_path):
        image=cv2.imread(img_path)
        grayscale_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
        return grayscale_image
    
    #downsampling the image using the imresize function which scales down the image by 0.1 factor
    @staticmethod
    def downsample_image(img_path):
        image = cv2.imread(img_path, 0)
        image = scipy.misc.imresize(image, 0.1)
        return image

        
    
        


