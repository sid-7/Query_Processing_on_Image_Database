# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:42:00 2019

@author: Siddharth
"""

from feature_store import FeatureStorage, ModelType


# Pickle the HOG
storage_path = "E:/ASU Courses/CSE 515- Multimedia and Web Databases/Project/Project Files/store/"
feature_storage = FeatureStorage(storage_path)
#feature_storage.store_to_pkl(ModelType.HOG)
feature_storage.store_to_pkl(ModelType.COLOR_MOMENTS)
