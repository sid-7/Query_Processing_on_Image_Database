# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:41:42 2019

@author: Siddharth
"""

# Implementation of Cosine similarity function
import numpy as np
from tqdm import tqdm




class SimilarityFunction:
    
    #providing model type at object creation
    def __init__(self, df, model_type):
        self.df = df
        self.model_type = model_type
    
    #extracting distances from json file    
    @staticmethod
    def extract_distance(json):
        try:
            return float(json["s"])
        except KeyError:
            return 0
    
    #core cosine implementation function    
    def get_similarity(self, image_id_1, image_id_2):
        dot_product=np.dot(self.df.loc[image_id_1,:], self.df.loc[image_id_2,:]) #taking dot product of two feature vectors
        magnitude=np.linalg.norm(self.df.loc[image_id_1,:])*np.linalg.norm(self.df.loc[image_id_2,:]) #calculating magnitude of each feature vector
        return (dot_product/magnitude) #returning the cosine similarity i.e. dot_product/magnitude
        
    #calculating similarity with each image and storing it in distances
    def get_image_distances(self, image_id):
        distances = []
        print("Calculating distances of images from image_id = " + str(image_id))
        #calculating distances to every other image except the provided image
        for other_image_id in tqdm([x for x in self.df.index.values if x != image_id]):
            print(np.shape(image_id))
            distances.append({"s": self.get_similarity(image_id, other_image_id), #appending the distance to distances
                              "other_image_id": other_image_id})
        distances.sort(key=self.extract_distance, reverse=True) #sorting the distances in decreasing order of similarity score
        return distances


