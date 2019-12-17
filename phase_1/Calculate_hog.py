# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:47:30 2019

@author: Siddharth
"""
#This file is used for calculating the histogram of gradients
import cv2
import numpy
import scipy
from Extraction import Extraction


class Calculate_hog(Extraction):
    
    
    #def __init__(self):
     #   Extraction.__init__(self)

    @staticmethod
    def get_hog_from_image(_image_path):
        image = cv2.imread(_image_path, 0)
        image = scipy.misc.imresize(image, 0.1)
        win_size = (64, 64)
        block_size = (8, 8)
        block_stride = (8, 8)
        cell_size = (2, 2)
        no_bins = 9
        deriv_aperture = 1
        win_sigma = 4.
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, no_bins, deriv_aperture, win_sigma,
                                histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)
    
        win_stride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hog.compute(image, win_stride, padding, locations)
        hist = hist.ravel()
    
        return hist
        