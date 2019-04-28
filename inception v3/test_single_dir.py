#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:22:11 2019

@author: shinong
"""
import sys,os
import tensorflow as tf
import shutil

def validated(pb_path, test_path):
    """
    """
    # step1. import pb
    with tf.gfile.FastGFile(pb_path, 'rb') as f:                                                                                      
        graph_def = tf.GraphDef()                                                                                                                  
        graph_def.ParseFromString(f.read())                                                                                                        
        _ = tf.import_graph_def(graph_def, name='') 

    # step2. 
    with tf.Session() as sess:
        acc=0
        wrong_dir='/home/shinong/Desktop/image10class/wrong_dir'
        for image in os.listdir(test_path):                                                                                                        
            if not image.endswith('.jpg'):                                                                                                         
                continue                                                                                                                           
                                                                                                                                          
            image_path = os.path.join(test_path, image)                                                                                            
            # Read in the image_data                                                                                                               
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()                                                                               
            # Feed the image_data as input to the graph and get first prediction                                                                   
            softmax_tensor = sess.graph.get_tensor_by_name('final_training_ops/final_ret:0')                                                                                                                                                 
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})                                                                          
            # Sort to show labels of first prediction in order of confidence                                                                       
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            
            # step3. pred_label --> true_label
#            print('[INFO]:image_paht:<{}> ---> out:<{}>'.format(image_path, top_k)) 
            if top_k[0] == 5:
                acc=acc+1   
            else:
                shutil.move(image_path,wrong_dir)
                print('[INFO]:image_paht:<{}> ---> out:<{}>'.format(image_path, top_k)) 
#            return
        accuracy=acc/len(os.listdir(test_path))    
        print(accuracy)
        
if __name__ == '__main__':
    pb_path = 'out1.pb'
#    image_dir = '/home/shinong/Desktop/image10class/马兰830/'
    image_dir=sys.argv[1]
    validated(pb_path, image_dir)
