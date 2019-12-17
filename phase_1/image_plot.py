# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:39:03 2019

@author: Siddharth
"""
#This file bascially contains the logic behind plotting the images
#we use matplotlib library for plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #image  module supports basic image loading, rescalling and displaying.


class ImagePlot:

    def __init__(self, image_id, image_base_path):
        self.image_id = image_id
        self.image_base_path = image_base_path

    def plot_comparison(self, other_image_ids):
        k = len(other_image_ids)
        fig = plt.figure(figsize=(k * 2, 1))
        columns = k + 1
        rows = 1

        image_ids = [self.image_id] + other_image_ids
        
        for i in range(1, columns * rows + 1):
            image_file_name = self.image_base_path + "Hand_" + str(image_ids[i - 1]).zfill(7) + ".jpg"
            img = mpimg.imread(image_file_name)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)

        return plt
