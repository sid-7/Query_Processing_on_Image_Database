# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 00:26:28 2019

@author: Siddharth
"""
#This file is used to set-up the console menu which would contains multiple options according to phase-1

from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
#the consolemenu library is used when you have to setup a menu based console
from Driver import *

menu = ConsoleMenu("CSE515 Phase 1 Console")

#Creating the do_extract_cm function which calls do_cm_create function from the driver
def do_extract_cm():
    do_cm_create(image_factory, cm_vector_mapping, feature_storage)

#Creating the do_extract_hog function which calls do_hog_create function from the driver
def do_extract_hog():
    do_hog_create(image_factory, hog_vector_mapping, feature_storage)

#This function is used to load the dataframe by selecting the model as an input
def do_load_df():
    model_type = raw_input("Select from [cm, hog]:")
    print("Selected: " + model_type)
    df = do_df_load(model_type, feature_storage)
    print(df.head)

#This functions calculates the distance of a given image according to the model and gives out k similar images
def do_get_distances():
    model_type = raw_input("Select from [cm, hog]: ")
    image_id = int(raw_input("Image ID: "))
    k = int(raw_input("K: "))
    print("Selected: " + model_type)
    similar_images = do_distances_get(model_type, image_id, k, feature_storage)
    plt=do_plot(image_id, similar_images) #passing image and similar images in do_plot function to display the images on the console
    save_plot(plt, image_id, {"model_type": model_type, "query_image_id": image_id, "k": k, "similar_images": similar_images})
    plt.show()

#setting up the menu items and asigning the related functions
extract_cm_item = FunctionItem("Extract Color Moments Features", do_extract_cm)
extract_hog_item = FunctionItem("Extract HOG Features", do_extract_hog)
load_df_item = FunctionItem("Load DF", do_load_df)
get_distances_item = FunctionItem("Get similar images", do_get_distances)

#appending the items on the menu
menu.append_item(extract_cm_item)
menu.append_item(extract_hog_item)
menu.append_item(load_df_item)
menu.append_item(get_distances_item)

menu.show() #displaying the menu on console using show()