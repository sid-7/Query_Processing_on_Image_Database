# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:20:59 2019

@author: Siddharth
"""

#this file is used to encode the numpy object into python INT to dump in json
import numpy
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        else:
            return super(MyEncoder, self).default(obj)