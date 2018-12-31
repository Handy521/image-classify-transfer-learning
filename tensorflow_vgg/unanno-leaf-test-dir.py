#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:38:47 2018

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
from collections import Counter

with open('labels-val-final') as f: 
    reader = csv.reader(f, delimiter='\n') 
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
with open('codes') as f: 
    codes = np.fromfile(f, dtype=np.float32) 
    codes = codes.reshape((len(labels), -1))

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)#one_hot


#test_dir='/media/shinong/ccfb3b62-5edd-437a-8a87-6d295fa6f144/single-accuracy/un-annotation/lack-Ca'
#batch=[]
#
#
flag=0
with tf.device('/cpu:0'):
    

    
        
    tf.reset_default_graph()
    with tf.Session() as sess:
        #tf.reset_default_graph()#清除默认图的堆栈，并设置全局图为默认图
       # p_keep_fc= tf.placeholder(tf.float32)
        inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
        #inputs_=tf.nn.dropout(inputs_,1.0)
        fc = tf.contrib.layers.fully_connected(inputs_, 256)
        #fc=tf.nn.dropout(fc,1.0)
        logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)#
 
        predicted = tf.nn.softmax(logits)
   
        saver = tf.train.Saver()#save and extract variable

        saver.restore(sess, tf.train.latest_checkpoint('leaf-final-checkpoints'))  
    
        feed = {inputs_: codes}       
        prediction = sess.run(predicted, feed_dict=feed).squeeze()
        #print(prediction)
        predic_list = prediction.tolist()
        a=[]
        for i in range(len(predic_list)):
            index = predic_list[i].index(max(predic_list[i]))
            if lb.classes_[index]!=labels[i]:
                print(lb.classes_[index]+":  "+labels[i])
                #print(lb.classes_[index]+":"+str(max(predic_list[i])))
                flag+=1
                a.append(''.join(str(labels[i])))
        
        print(flag/i)
        print(i)        
ret1=Counter(a)
ret2=Counter(labels)
for k1,v1 in ret1.items():
    for k2,v2 in ret2.items():
        if k1==k2:
            print(str(v2)+','+k1+','+str((v2-v1)/v2))        
        
        
