# -*- coding: utf-8 -*-
"""
Created on Mon Oct 01 15:55:51 2018

@author: yong2
"""
import numpy as np
import cv2
import os
#import numpy.random as npr
#from utils import IoU
anno_file = "sunburn.txt"
pos_save_dir = "sunburn"
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print ("%d pics in total" % num)
idx = 0

p_idx=0
for annotation in annotations:
    annotation = annotation.strip().split('.')
    im_path = '/media/shinong/ccfb3b62-5edd-437a-8a87-6d295fa6f144/single-accuracy/12..03-12.04--标注/柑橘-日灼-果实/'\
                +annotation[0]+'.jpg'
    annotation2=annotation[1].strip().split(' ')
    bbox = list(map(float, annotation2[0:])) #map在python2和3中不同
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    
    img = cv2.imread(im_path)
    if str(img)=='None':
        im_path = im_path[0:-3]+'png'
        img = cv2.imread(im_path)
        if str(img)=='None':
            im_path = im_path[0:-3]+'JPG'
            img = cv2.imread(im_path)
            if str(img)=='None':
                im_path = im_path[0:-3]+'jpeg'
                img = cv2.imread(im_path)
                if str(img)=='None':
                    continue
    idx += 1
    if idx % 100 == 0:
        print (idx, "images done")

    height, width, channel = img.shape    
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
   
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        cropped_im = img[int(y1) : int(y2), int(x1) : int(x2), :]
        resized_im = cv2.resize(cropped_im, (299, 299), interpolation=cv2.INTER_LINEAR)
        xImg=cv2.flip(resized_im,1,dst=None)     
        save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)           
        cv2.imwrite(save_file, resized_im)
        p_idx += 1
        save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)          
        cv2.imwrite(save_file,  xImg)
        p_idx += 1


#f1.close()
#f3.close()
