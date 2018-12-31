#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:59:04 2018

@author: shinong
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.platform import gfile
INCEPTION_MODEL_FILE = './tensorflow_inception_graph.pb'
CHECKPOINT_DIR = './fruit2' 
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称
flower_dict={0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'} # 读取数据 

checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
path='/home/shinong/big_bumper/tensorflow-vgg/flower_photos/sunflowers/44079668_34dfee3da1_n.jpg'
image_data = tf.gfile.FastGFile(path, 'rb').read() # 评估 
y_test = [3]
with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        with gfile.FastGFile(os.path.join(INCEPTION_MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    # 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME]) 
        
        bottleneck_values = sess.run(bottleneck_tensor,{jpeg_data_tensor: image_data})
        bottleneck_values = [np.squeeze(bottleneck_values)]
        
#        saver = tf.train.import_meta_graph('/home/shinong/big_bumper/1213-20Kphoto/fruit/orange.meta') 
#        saver.restore(sess, tf.train.latest_checkpoint('/home/shinong/big_bumper/1213-20Kphoto/fruit/'))
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name(
            'BottleneckInputPlaceholder').outputs[0]
        predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]
        all_predictions = []
        all_predictions = sess.run(predictions, {input_x: bottleneck_values})
        index=str(all_predictions)[1]
        index=int(index)
        print(path+' '+'预测为：'+flower_dict[index])
        if y_test is not None: 
            correct_predictions = float(sum(all_predictions == y_test)) 
            print('\nTotal number of test examples: {}'.format(len(y_test))) 
            print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))

        