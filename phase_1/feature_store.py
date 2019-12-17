# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:16:19 2019

@author: Siddharth
"""

import json

import pandas as pd
from enum import Enum


class ModelType(Enum):
    COLOR_MOMENTS = "cm"
    HOG = "hog"

class FeatureStorage:

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.switcher_json = {
            ModelType.COLOR_MOMENTS: "color_moments.json",
            ModelType.HOG: "hog.json",
        }

        self.switcher_pkl = {
            ModelType.COLOR_MOMENTS: "color_moments.pkl",
            ModelType.HOG: "hog.pkl",
        }

        self.switcher_distances = {
            ModelType.COLOR_MOMENTS: "color_moments.distances.json",
            ModelType.HOG: "hog.distances.json",
        }
            
    def clear_model_storage(self, model_type):
        open(self.storage_path + self.switcher_json.get(model_type), "w").close()
    
    #storing the feature set in the json format
    def store_feature_set(self, model_type, feature_vectors, json_only=False):
        db_file = self.storage_path + self.switcher_json.get(model_type)
        with open(db_file, 'w') as f:
            json.dump(feature_vectors, f)
        if not json_only:
            self.store_to_pkl(model_type)
    
    #loading the dataframe according to the model        
    def load_to_df(self, model_type):
        print("1" + self.storage_path)
        if model_type == "cm":
            db_file = self.storage_path + self.switcher_pkl.get(ModelType.COLOR_MOMENTS)
        elif model_type== "hog":
            db_file = self.storage_path + self.switcher_pkl.get(ModelType.HOG)
        df = pd.read_pickle(db_file)
        return df
    
    #storing the file in pickle format for faster retrivel 
    def store_to_pkl(self, model_type):
        db_file = self.storage_path + self.switcher_json.get(model_type)
        df = pd.read_json(db_file)
        df = df.T
        df.to_pickle(self.storage_path + self.switcher_pkl.get(model_type))
        
    #function to store the distances    
    def store_distances(self, model_type, distances_json):
        db_file = self.storage_path + self.switcher_json.get(model_type)
        with open(db_file, 'w') as f:
            json.dump(distances_json, f)