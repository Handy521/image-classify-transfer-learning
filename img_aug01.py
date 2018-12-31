#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:30:53 2018

@author: shinong
"""


import numpy as np
import cv2
import os
#import numpy.random as npr
#from utils import IoU

pos_save_dir = "photo-aug"

first_dir = os.listdir(pos_save_dir)

idx = 0

p_idx=0
for second_dir in first_dir:
    all_im_path=os.listdir(pos_save_dir+'/'+second_dir)
    for ii,im_path in enumerate(all_im_path):
        
  
    
        img = cv2.imread(pos_save_dir+'/'+second_dir+'/'+im_path)
    
        idx += 1
        if idx % 100 == 0:
            print (idx, "images done")
          
        xImg=cv2.flip(img,1,dst=None)     
        save_file = os.path.join(pos_save_dir+'/'+second_dir, "1%s.jpg"%p_idx)           
        cv2.imwrite(save_file, xImg)        
        p_idx += 1
        xImg2=cv2.flip(img,0,dst=None)
        save_file = os.path.join(pos_save_dir+'/'+second_dir, "0%s.jpg"%p_idx)          
        cv2.imwrite(save_file,  xImg2)
        p_idx += 1


#f1.close()
#f3.close()
