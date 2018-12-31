#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:30:12 2018

@author: shinong
"""

import os
import numpy as np
import random
import shutil
image_dir='./photo'
files=os.listdir(image_dir)
result=random.sample(range(1,19451),1500)#generate 1500 random number for test set
result.sort()#sort
image={}#dict
for ii,file in enumerate(files):
    b=os.path.join(image_dir,files[ii])
    c=os.listdir(b)
#    a.append(c)
    for cc in c:
        image[cc]=files[ii]#all img dict{key:value}={imgname:class}
move_img=[]        
for k,v in image.items():
    move_img.append(k)#extract image name

need_img=[]
for j in result:
    need_img.append(move_img[j])#random test set image 
classes={}    
for k,v in image.items():
    for jj in need_img:
        if k==jj:#match imgname 
          classes[jj]=v #get test image class 
for k,v in classes.items():#new dir save image
    if not os.path.exists(v):
        os.mkdir(v)
    path=os.path.join(image_dir,v,k)
    if os.path.isfile(path):
        shutil.move(path,v)
