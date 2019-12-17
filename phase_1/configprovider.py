# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:27:05 2019

@author: Siddharth
"""

#for configuring paths.
import json


class ConfigProvider:

    def __init__(self, config_path):
        with open(config_path) as json_file:
            self.config = json.load(json_file)

    def get_image_base_path(self):
        return self.config.get("image_base_path")

    def get_storage_path(self):
        return self.config.get("storage_path")

    def get_output_path(self):
        return self.config.get("output_path")