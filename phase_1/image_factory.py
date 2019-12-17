# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:37:08 2019

@author: Siddharth
"""
#This file if basically used for preparing the fetching the image from the dataset
#glob is a pathname matching module
import glob
#regular expression matching library
import re


class ImageFactory:

    def __init__(self, base_path):
        self.base_path = base_path
        self.image_type = ".jpg"

    def get_image_path_list(self): #fetching the image from list of paths
        _image_paths = [_image_path for _image_path in glob.glob(self.base_path + "/*" + self.image_type)]
        return _image_paths

    @staticmethod
    def get_image_id_from_path(_image_path):
        return re.findall(r'\d+', str(_image_path).split("/Hands/")[1].split(".jpg")[0])[0] #using reggualr expression for fetching the image_id from the image path.