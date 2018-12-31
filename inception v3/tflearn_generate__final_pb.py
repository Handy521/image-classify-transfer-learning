#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:17:06 2018

@author: shinong
"""

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提出的Inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

# 下载的谷歌训练好的Inception-v3模型文件目录
#MODEL_DIR = 'model/'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中，免去重复的计算。
# 下面的变量定义了这些文件的存放地址。
CACHE_DIR = 'label-tmp-flower/'

# 图片数据文件夹。
# 在这个文件夹中每一个子文件夹代表一个需要区分的类别，每个子文件夹中存放了对应类别的图片。
# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 1
BATCH = 1
#LEARNING_RATE = 0.008
#STEPS = 8000
#BATCH = 100


def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Session() as sess:
        with gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def main():
    # 读取所有图片。
    n_classes = 5
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()
    #合并成一个pb文件——with-default
    bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')

    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 1024], stddev=0.001))
        biases = tf.Variable(tf.zeros([1024]))
        logits1 = tf.matmul(bottleneck_input, weights) + biases

        weights1 = tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.001))
        biases1 = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(logits1, weights1) + biases1
        final_tensor = tf.nn.softmax(logits, name='final_ret')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['final_training_ops/final_ret'])
        with gfile.FastGFile('out.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())



if __name__ == '__main__':
    main()
