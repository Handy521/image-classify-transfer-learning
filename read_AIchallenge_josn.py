#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:37:29 2018

@author: shinong
"""

import tensorflow as tf
from data_provider import provide
import numpy as np
import shutil
import os
import json
def provide(annotation_path=None, images_dir=None):
    """Return image_paths and class labels.
    
    Args:
        annotation_path: Path to an anotation's .json file.
        images_dir: Path to images directory.
            
    Returns:
        image_files: A list containing the paths of images.
        annotation_dict: A dictionary containing the class labels of each 
            image.
            
    Raises:
        ValueError: If annotation_path does not exist.
    """
    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')
        
    annotation_json = open(annotation_path, 'r')
    annotation_list = json.load(annotation_json)
    
    image_files = []
    annotation_dict = {}
    
    for d in annotation_list:
        image_name = d.get('image_id')
        disease_class = d.get('disease_class')
        
        if images_dir is not None:
            image_name = os.path.join(images_dir, image_name)
        image_files.append(image_name)
        annotation_dict[image_name] = disease_class
    return image_files, annotation_dict
image_dir='/media/shinong/study/ChallengeAI_Agriculture/AgriculturalDisease_validationset/images'
annotation='/media/shinong/study/ChallengeAI_Agriculture/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'
a ,b = provide(annotation,image_dir)

classs = []
for k,v in b.items():
    classs.append(v)
#print(set(classs))

for i in range(len(set(classs))):
    
    for k ,v in b.items():
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        if v==i:
            shutil.copy(k,str(i))
#        path=os.path.join('./data',str(i))
#        if v==i:
#            shutil.move(k,path)

