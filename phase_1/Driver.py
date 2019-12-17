# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:43:36 2019

@author: Siddharth
"""

#Driver file works as an intermediate which connects the menu with the core functions
import numpy as np
import os
from tqdm import tqdm #for displaying the progress bar
import json
from MyEncoder import MyEncoder
from configprovider import ConfigProvider
from similarity_function import SimilarityFunction
from image_factory import ImageFactory
import Calculate_color_moments, Calculate_hog
from Calculate_color_moments import Calculate_color_moments
from Calculate_hog import Calculate_hog
from image_plot import ImagePlot
from feature_store import FeatureStorage, ModelType

#dataset path
#feature storage path

config_path = "D:/ASU Courses/CSE-515 Multimedia and Web Databases/Project/cse515-mwdb-project/phase_1/config.json"
config_provider = ConfigProvider(config_path)
image_base_path = config_provider.get_image_base_path()
storage_path = config_provider.get_storage_path()
output_path = config_provider.get_output_path()



image_factory = ImageFactory(image_base_path)
cm_vector_mapping = Calculate_color_moments()
hog_vector_mapping = Calculate_hog()
feature_storage = FeatureStorage(storage_path)
image_paths = image_factory.get_image_path_list()

#for creating the color moments
#this function iterates over the whole dataset and stores the feature set in .json format
def do_cm_create(_image_factory, _cm_vector_mapping, _feature_storage):
    print("Creating Color Moments features:")
    cm_feature_vectors={}
    #here tqdm helps displaying the progress bar as the loop moves forward
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        cm_from_image = _cm_vector_mapping.get_color_moments_from_image(_image_path)
        cm_feature_vectors[_image_id] = cm_from_image.tolist()

    _feature_storage.clear_model_storage(ModelType.COLOR_MOMENTS)
    _feature_storage.store_feature_set(ModelType.COLOR_MOMENTS, cm_feature_vectors)
    print("Features stored in " + storage_path + "color_moments.json")


#for creating the color moments
#this function iterates over the whole dataset and stores the feature set in .json format
def do_hog_create(_image_factory, _hog_vector_mapping, _feature_storage):
    print("Creating HOG features:")
    hog_feature_vectors = {}
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        hog_from_image = _hog_vector_mapping.get_hog_from_image(_image_path)
        hog_feature_vectors[_image_id] = hog_from_image.tolist()

    _feature_storage.clear_model_storage(ModelType.HOG)
    _feature_storage.store_feature_set(ModelType.HOG, hog_feature_vectors)
    print("Features stored in " + storage_path + "hog.json")


def do_df_load(model_type, _feature_storage):
    return _feature_storage.load_to_df(model_type)

#for calculating the distances from the image
def do_distances_get(model_type, image_id, k, _feature_storage):
    df = feature_storage.load_to_df(model_type)    
    similarity_function = SimilarityFunction(df, model_type) #called the similarity function which calculates the distances
    distances_json = similarity_function.get_image_distances(image_id)[:k]
    return distances_json

#for plotting the similar images on to the console
def do_plot(image_id, distances_json):
    image_plot = ImagePlot(image_id, image_base_path)
    other_image_ids = [node["other_image_id"] for node in distances_json]
    print("other")
    print(other_image_ids)
    return image_plot.plot_comparison(other_image_ids)

def save_plot(plt, image_id, query_info):
    output_directory = output_path + "/query_image_" + str(image_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_image_path = output_directory + "/output.png"
    output_info_path = output_directory + "/query.json"
    print(output_image_path)
    plt.savefig(output_image_path)
    print(query_info)
    with open(output_info_path, 'w') as f:
        json.dump(query_info,f, cls=MyEncoder)