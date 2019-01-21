#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:14:06 2019

@author: shinong
"""

import tensorflow as tf
import numpy as np
import os
import time
import cv2
import shutil
from collections import Counter
import slim.nets.resnet_v2 as resnet_v2
import tensorflow.contrib.slim as slim
def decorate(func):
    def call_back(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print('[INFO]:此次操作耗时<{:.2f}>秒'.format(end - start))
        return ret

    return call_back
def load_img(path):
    img= cv2.imread(path)
#    img = img / 255.0
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = cv2.resize(crop_img, (224, 224))
#    resized_img = skimage.transform.resize(crop_img, (224, 224))
    # bgr --> rgb 
    bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # rgb normalizer
    ret_img = (bgr_img/255 - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    
    return ret_img


def test():
    
#    image_set=load_data2('/home/shinong/Desktop/validation/3')
    image_set='/home/shinong/Desktop/5class/蒲公英14'
    with tf.name_scope('init_data'):
        # 定义input_images为图片数据
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3],
                                      name='input_images')
        # 定义input_labels为labels数据
        input_labels = tf.placeholder(dtype=tf.int32, shape=[None, 23], name='input_labels')

        # 定义dropout
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        is_training = tf.placeholder(tf.bool, name='is_training')
#    Y = tf.placeholder("float", [None, 10]) 

    # Define the model:
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_18(inputs=input_images, num_classes=23,
                                                      is_training=is_training)

#    logits, end_points = resnet_v2.resnet_v2_18(inputs=X_input, num_classes=10,
#                                                      is_training=False)
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
#    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#    train_op = tf.train.AdadeltaOptimizer(0.001, 0.95).minimize(cost)
#    train_op =tf.train.GradientDescentOptimizer(0.005).minimize(cost)
    predict=[]
    acc=0
    c=len(os.listdir(image_set))
    wrong_dir='/home/shinong/Desktop/wrong_dir'
    predict_op = tf.argmax(logits, 1)
#    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(Y, 1)), tf.float32))
    saver = tf.train.Saver()  
    with tf.Session() as sess:
        saver.restore(sess, './models_v3_res18/model.ckpt-284000')  
#        for batch in range(len(image_set),64):
        
        for image in os.listdir(image_set):
            image=os.path.join(image_set,image)
            img=load_img(image)
            img=img.reshape(1,224,224,3)
        
            pred=sess.run(predict_op,feed_dict={input_images: img,is_training:False})
            predict.append(pred[0])
            if pred[0]== 14:
                acc=acc+1   
            else:
                shutil.move(image,wrong_dir)
                print('[INFO]:image_paht:<{}> ---> out:<{}>'.format(image, pred[0])) 
        print(acc/c ) 
        print(Counter(predict))
        
            
#            print('[INFO]:gt:<yes> ---> out:<{}>'.format(pred)) 
#            start_time = time.time()
#        accuracy=acc/c  
           
#        b=Counter(predict)
#        b = collections.Counter(predict)
        
        
##        print(b,len(pred),(b[0]/len(pred))*100)
#    v=[]
#    for k1,v1 in b.items():
#        v.append(v1)
#    print(max(v)/len(pred))    
           
if __name__=='__main__':
    test()