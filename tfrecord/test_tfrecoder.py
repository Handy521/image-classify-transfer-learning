# -*- coding: utf-8 -*-


 
import tensorflow as tf
import numpy as np 
import pdb
from datetime import datetime
from VGG16 import *

 
batch_size = 32
lr = 0.0001
n_cls = 11
max_steps = 30000
 
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
    img = tf.reshape(img, [480, 120, 3])
    img = tf.image.resize_images(img, [224,224],method=0)
    
    # 转换为float32类型，并做归一化处理
    img = tf.image.convert_image_dtype(img,dtype = tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    return img, label

 
def train_test():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')    
    
    keep_prob = tf.placeholder(tf.float32)
    
    output = vgg16(x, keep_prob, n_cls, f = True)
    
#    probs = tf.nn.softmax(output)
 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=y))
#    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))
    
#    train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
#    train_step = tf.train.AdamOptimizer(0.005).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
 
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(y, 1)), tf.float32))
    
    train_images, train_labels = read_and_decode('./train.tfrecords')
    
    test_images, test_labels = read_and_decode('./test_xx.tfrecords')
    
    train_img_batch, train_label_batch = tf.train.shuffle_batch([train_images, train_labels],
                                                                batch_size=batch_size,
                                                                capacity=400,
                                                                min_after_dequeue=150,
                                                                num_threads = 16
                                                                )
    
    
    test_img_batch, test_label_batch = tf.train.shuffle_batch([test_images, test_labels],
                                                                batch_size=batch_size,
                                                                capacity=400,
                                                                min_after_dequeue=150,
                                                                num_threads = 16
                                                            )
    
    train_label_batch = tf.one_hot(train_label_batch, 11, 1, 0)                                                  
    test_label_batch = tf.one_hot(test_label_batch, 11, 1, 0)
    
       
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    saver = tf.train.Saver()  

    with tf.Session() as sess:
        sess.run(init)
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
        # train start
        for i in range(max_steps):
            batch_x, batch_y = sess.run([train_img_batch, train_label_batch])
            _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
            
            if i % 100 == 0:
                train_arr = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
                print("%s: Step [%d]  Loss : %f, training accuracy :  %g" % (datetime.now(), i, loss_val, train_arr))
                # 只指定了训练结束后保存模型，可以修改为每迭代多少次后保存模型
            
            if (i + 1) == max_steps : 
                saver.save(sess, './model/model.ckpt')
        
        total_acc = 0
      
        test_x, test_y = sess.run([test_img_batch, test_label_batch])
        total_acc += sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0})
        print("accuracy:",sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0}))

        
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    train_test()
    
    
    
    
