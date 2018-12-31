#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:30:18 2018

@author: xiongsirui
"""

import os 
import tensorflow as tf 
from PIL import Image  
import numpy as np
import cv2

data_dir = './train-data/'
cwd = os.getcwd()
print(cwd)
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]



def load_img(path):
    img = cv2.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = cv2.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img

writer= tf.python_io.TFRecordWriter("train.tfrecords") #要生成的文件
 
for index,name in enumerate(classes):
    print(str(index) + "-->" + name)
    class_path=cwd+'/train-data'+'/'+ name +'/'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name 
        
        print(img_path)
#for each in classes:
#    print("Starting {} images".format(each))
#    class_path = data_dir + each
#    files = os.listdir(class_path)
#    for ii, file in enumerate(files, 1):
            # Add images to the current batch
            # utils.load_image crops the input images for us, from the center
              
        img = load_img(os.path.join(img_path))
    
        
        
        
        img_raw=img.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) 
        writer.write(example.SerializeToString())
    
        
writer.close()



