#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:24:10 2018

@author: shinong
"""

import os
import shutil
pwd = os.getcwd()
contents = os.listdir('./photo')
classes = [each for each in contents if os.path.isdir( './photo/'+each)]
#for each in classes:
for ii,each in enumerate(classes):    
    print("Starting {} images".format(each))
    class_path = './photo/'+each
    files = os.listdir(class_path)
#    if len(files)<200:
#        shutil.rmtree(each)
#    if not os.path.exists(str(ii)):
#        os.mkdir(str(ii))
#    shutil.copytree(class_path,str(ii))                 
    os.rename(class_path,str(ii))