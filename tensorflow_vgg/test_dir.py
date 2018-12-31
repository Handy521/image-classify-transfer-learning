#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:42:33 2018

@author: shinong
"""

import tensorflow as tf
import numpy as np
import vgg16
import utils
import matplotlib.pyplot as plt
from scipy.ndimage import imread
import cv2
import csv
import os


with open('labels-leaf2') as f: 
    reader = csv.reader(f, delimiter='\n') 
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)#one_hot


test_dir='./lack-N'
batch=[]

with tf.device('/cpu:0'):
    
    with tf.Session() as sess:
        

        #input test data in model with vgg16.npy to build the layer relu 6 
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        
        vgg = vgg16.Vgg16()          
        #need vgg16.npy
        vgg.build(input_)
#              
        
        #read test-data a dir
        files=os.listdir(test_dir)
        for file in files:
            img = utils.load_image(os.path.join(test_dir, file))
            batch.append(img.reshape((1, 224, 224, 3)))
        
        images=np.concatenate(batch)
        feed_dict = {input_: images}     
        code = sess.run(vgg.relu6, feed_dict=feed_dict)
    
        
    
    
#        tf.reset_default_graph()#清除默认图的堆栈，并设置全局图为默认图

        inputs_ = tf.placeholder(tf.float32, shape=[None, code.shape[1]])

        fc = tf.contrib.layers.fully_connected(inputs_, 256)

        logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)#
 
        predicted = tf.nn.softmax(logits)
   
        saver = tf.train.Saver()#save and extract variable

        saver.restore(sess, tf.train.latest_checkpoint('leaf-checkpoints'))  
    
        feed = {inputs_: code}       
        prediction = sess.run(predicted, feed_dict=feed).squeeze()
        #print(prediction)
        predic_list = prediction.tolist()
        
        for i in range(len(predic_list)):
            index = predic_list[i].index(max(predic_list[i]))
            print(lb.classes_[index]+":"+str(max(predic_list[i])))
        
        
        
        
