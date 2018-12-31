"""
This is a simple example of transfer learning using VGG.
Fine tune a CNN from a classifier to regressor.
Generate some fake data for describing cat and tiger length.

Fake length setting:
Cat - Normal distribution (40, 8)
Tiger - Normal distribution (100, 30)

The VGG model and parameters are adopted from:
https://github.com/machrisaa/tensorflow-vgg

Learn more, visit my tutorial site: [莫烦Python](https://morvanzhou.github.io)
"""

from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import csv
data_dir = './val-data/'


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
 
    
    # 转换为float32类型，并做归一化处理
    img = tf.image.convert_image_dtype(img,dtype = tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    return img, label


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img


def load_data():
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]
    batch=[]
    labels=[]
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # Add images to the current batch
            # utils.load_image crops the input images for us, from the center
            img = load_img(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
    return batch,labels
#    imgs = {'tiger': [], 'kittycat': []}
#    for k in imgs.keys():
#        dir = './for_transfer_learning/data/' + k
#        for file in os.listdir(dir):
#            if not file.lower().endswith('.jpg'):
#                continue
#            try:
#                resized_img = load_img(os.path.join(dir, file))
#            except OSError:
#                continue
#            imgs[k].append(resized_img)    # [1, height, width, depth] * n
#            if len(imgs[k]) == 816:        # only use 400 imgs to reduce my memory load
#                break
    # fake length data for tiger and cat
#    tigers_y = np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100)
#    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
#    tigers_y=np.zeros(814)
#    cat_y=tigers_y+1
    #return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None,14])
        self.p_keep_fc = tf.placeholder(tf.float32)
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
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
        self.fc6 = tf.layers.dense(self.flatten, 4096, tf.nn.relu, name='fc6')
        self.fc6=tf.nn.dropout(self.fc6,self.p_keep_fc)
        self.fc7 = tf.layers.dense(self.fc6, 256, tf.nn.relu, name='fc7')
        self.fc7=tf.nn.dropout(self.fc7,self.p_keep_fc)
        self.out = tf.layers.dense(self.fc7, 14, name='out')
        
        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        
        else:   # training graph
            #self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tfy, logits=self.out))
            self.correct_pred = tf.equal(tf.argmax(self.out,1), tf.argmax(self.tfy,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            #self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.train_op = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y,self.p_keep_fc:0.5})
        return loss

    def predict(self, val_x,val_y,drop):
#        fig, axs = plt.subplots(1, 2)
#        for i, path in enumerate(paths):
#            x = load_img(path)
        #tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        feed = {self.tfx: val_x,
                self.tfy: val_y,self.p_keep_fc:drop
                }
        val_acc = self.sess.run(self.accuracy, feed_dict=feed)
        #out_value = self.sess.run(self.out, {self.tfx: x})
        return val_acc

    def save(self, path='./for_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)
        
def get_batches(x, y, n_batches=300):
    """ 这是一个生成器函数，按照n_batches的大小将数据划分了小块 """
    batch_size = len(x)//n_batches
    
    for ii in range(0, n_batches*batch_size, batch_size):
        # 如果不是最后一个batch，那么这个batch中应该有batch_size个数据
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        # 否则的话，那剩余的不够batch_size的数据都凑入到一个batch中
        else:
            X, Y = x[ii:], y[ii:]
        # 生成器语法，返回X和Y
        yield X, Y

#contents = os.listdir(data_dir)
#classes = [each for each in contents if os.path.isdir(data_dir + each)]
#batch=[]
#labels=[]
#for each in classes:
#    print("Starting {} images".format(each))
#    class_path = data_dir + each
#    files = os.listdir(class_path)
#    for ii, file in enumerate(files, 1):
#        # Add images to the current batch
#        # utils.load_image crops the input images for us, from the center
#        img = load_img(os.path.join(class_path, file))
#        batch.append(img.reshape((1, 224, 224, 3)))
#        labels.append(each)
#np.save('batch-val.npy',batch)
#np.save('labels-val.npy',labels)
#print('year')

batch=np.load('batch.npy')
labels=np.load('labels.npy')
batch2=np.load('batch-val.npy')
labels2=np.load('labels-val.npy')
xs=np.concatenate(batch)    
val=np.concatenate(batch2)  
lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)  
lb.fit(labels2)
labels_val = lb.transform(labels2)  
vgg = Vgg16(vgg16_npy_path='./vgg16.npy')
print('Net built')
#train_loss = vgg.train(train_img_batch, train_label_batch)
epochs=100
#    
#    
for i in range(epochs):
    for x,y in get_batches(xs,labels_vecs,300):
        #b_idx = np.random.randint(0, len(xs), 100)
        train_loss = vgg.train(x,y)
        #print(i, 'train loss: ', train_loss)
#            print( i,'train loss: ', train_loss)
    print(i, 'train loss: ', train_loss)
    test_indices = np.arange(len(val)) # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:181]
    
#    print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
#                sess.run(predict_op, feed_dict={X: teX[test_indices],
      
#    for x2,y2 in get_batches(val,labels_val,3):
    out_value=vgg.predict(val[test_indices],labels_val[test_indices],1.0)    
    print( i,'accuracy: ', out_value )
vgg.save('./model/leaf_learn')      # save learned fc layers

