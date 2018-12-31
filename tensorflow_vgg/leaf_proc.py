#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:23:10 2018

@author: shinong
"""

import os

import numpy as np
import tensorflow as tf

import vgg16
import utils
import cv2
#data_dir = '/media/shinong/ccfb3b62-5edd-437a-8a87-6d295fa6f144/single-accuracy/un-annotation/'
data_dir = '/home/shinong/big_bumper/leaf-lack-elememt-unlabel/val-data/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

#将图像批量batches通过VGG模型，将输出作为新的输入：

# Set the batch size higher if you can fit in in your GPU memory
batch_size = 10
codes_list = []
labels = []
batch = []

codes = None

with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)

    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # Add images to the current batch
            # utils.load_image crops the input images for us, from the center
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))            
            labels.append(each)
            
#            src=cv2.imread(os.path.join(class_path, file))
#            xImg = cv2.flip(src,1,dst=None) #水平镜像
#            img2 = utils.load_image2(xImg)
#            batch.append(img2.reshape((1, 224, 224, 3)))            
#            labels.append(each)
            # Running the batch through the network to get the codes
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)

                feed_dict = {input_: images}
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
                
                # Here I'm building an array of the codes
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))
                
                # Reset to start building the next batch
                batch = []
                print('{} images processed'.format(ii))
with open('codes', 'w') as f:
    codes.tofile(f)
    
import csv
with open('labels-val-final', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)

