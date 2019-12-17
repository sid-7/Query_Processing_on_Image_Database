# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:47:04 2019

@author: Siddharth
"""
#This file contains class and functions for calculating the color moments from the image

import cv2
import numpy as np
from scipy import stats
from Extraction import Extraction


class Calculate_color_moments(Extraction):
    
    
    def __init__(self):
        Extraction.__init__(self)

    #this function extracts color moments pre window and is called from the get_color_moments_from_image function
    @staticmethod
    def get_color_moments_from_image_window(_image_window_yuv):
        image_window_y, image_window_u, image_window_v = cv2.split(_image_window_yuv)
        image_window_color_moments = np.ravel(
            [[np.mean(channel), np.math.sqrt(np.var(channel)), stats.skew(channel.ravel())] for channel in
             [image_window_y, image_window_u, image_window_v]])
        # here I have used numpy library function ravel which is used to flatten the array
        return image_window_color_moments # this function returns the color_moments of the particular input window

    #this function merges the color_moments per window and then returns color_moments of the whole image
    def get_color_moments_from_image(self, _image_path):
        _image_yuv = Extraction.convert_to_yuv(_image_path)
        image_windows_yuv = Extraction.split_image_into_windows(_image_yuv)
        _cm_image = np.ravel(
            [self.get_color_moments_from_image_window(image_window_yuv) for image_window_yuv in image_windows_yuv]
        )
        #the _cm_image contains the color moments for the whole image
        return _cm_image