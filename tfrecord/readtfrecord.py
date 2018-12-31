#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import csv
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
 
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.int64)
                                       })
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.image.resize_images(img, [224,224],method=0)
    
    # 转换为float32类型，并做归一化处理
    img = tf.image.convert_image_dtype(img,dtype = tf.float32)
    label = tf.cast(features['label'], tf.int32)    
    return img, label
class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        tfy = tf.placeholder(tf.float32, [None,14])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        fc = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc',reuse=tf.AUTO_REUSE)
        out = tf.layers.dense(fc, 14, name='out',reuse=tf.AUTO_REUSE)
        loss = tf.losses.mean_squared_error(labels=tfy, predictions=out)
        #tf.get_variable_scope().reuse_variables()
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        run_options=tf.RunOptions(timeout_in_ms=10000)
        train_images, train_labels = read_and_decode('./train.tfrecords')
        train_img_batch, train_label_batch = tf.train.shuffle_batch([train_images, train_labels],
                                                                batch_size=16,
                                                                capacity=150,
                                                                min_after_dequeue=50,
                                                                num_threads = 2
                                                                )
        #tf.reset_default_graph() 
        train_label_batch = tf.one_hot(train_label_batch, 14, 1, 0)  
        
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer(),options=run_options)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord,start=True)
                
            # train start
            for i in range(3000):
                batch_x, batch_y = sess.run([train_img_batch, train_label_batch])
                _, loss_val = sess.run([loss, train_op], feed_dict={tfx:batch_x, tfy:batch_y},options=run_options)
                print( i,'train loss: ', loss_val)           
            coord.request_stop()
            coord.join(threads)
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout
    
vgg = Vgg16(vgg16_npy_path='./vgg16.npy')
print('Net built')




