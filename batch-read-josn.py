#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 10:28:31 2018

@author: shinong
"""

import json
import os
import shutil
import cv2
import numpy as np
#大文件夹路径，里面有图片和一个与图片名相对应的josn文件夹outputs
anno="/media/shinong/ccfb3b62-5edd-437a-8a87-6d295fa6f144/测试照片11.27-12.04"

#def main():
idx=0
#p_idx=1743
p_idx=9000#当程序中断时，从该位置继续
annotations=os.listdir(anno)  
#for ii,imgdir in enumerate(annotations):
for ii in range(len(annotations)):
    jos_path=os.path.join(anno, annotations[ii],'outputs')#josn文件路径
    josn_file=os.listdir(jos_path)
    for file in josn_file:
        idx += 1
        if idx>9000:
            annotation_json = open(os.path.join(jos_path, file), 'r')
            annotation_list = json.load(annotation_json)
            bb=[]
            try:
                stra=annotation_list['outputs']['object'] 
            except:
                continue                      
            for num in range(len(stra)):
                b=(int(stra[num]['bndbox']['xmin']),int(stra[num]['bndbox']['ymin']),
                   int(stra[num]['bndbox']['xmax']),int(stra[num]['bndbox']['ymax']))
                bb=(" ".join([str(a) for a in b])+" ")
            try:
                cc=bb.strip().split(' ')
            except:
                continue
            bbox = list(map(float, cc)) 
            boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
            image_name = file[0:-4]+'jpg'#josn文件名和图片名一致
       # if images_dir is not None:
            image_name = os.path.join(jos_path[0:-7], image_name)
            
            img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            
            #idx += 1
            if idx % 100 == 0:
                print (idx, "images done")
            for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
                x1, y1, x2, y2 = box
                w = x2 - x1 + 1
                h = y2 - y1 + 1
           
                if x1 < 0 or y1 < 0 or min(w,h)<40:
                    #x1=0
                    continue
                 
                try:    
                    cropped_im = img[int(y1) : int(y2), int(x1) : int(x2), :]#裁剪出图片
                    resized_im = cv2.resize(cropped_im, (299, 299), interpolation=cv2.INTER_LINEAR)
                except :
               # print(image_name)
                    continue
                
                if not os.path.exists(annotations[ii]):#有多少个类别就在当前目录下生成相应文件夹0-30
                    os.mkdir(annotations[ii])
                save_file = os.path.join(annotations[ii], "%s.jpg"%p_idx)           
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

#        for d in annotation_list:
#            image_name = d.get('image_id')
#            disease_class = d.get('disease_class')
#            
#            if images_dir is not None:
#                image_name = os.path.join(images_dir, image_name)
#            image_files.append(image_name)
#            annotation_dict[image_name] = disease_class
#if __name__=='__main__':
#    main()
