# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/'):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    
    img = np.array(img)
    filename, file_ext = os.path.splitext(os.path.basename(img_file))
    res_img_file = dirname + 'res_' + filename + '.jpg'

    for i, box in enumerate(boxes):
        # poly = np.array(box).astype(np.int32).reshape((-1))
        # strResult = ','.join([str(p) for p in poly]) + '\r\n'
        # f.write(strResult)
        # poly = poly.reshape(-1, 2)                
        # poly = poly.reshape((-1, 1, 2))

        # poly = np.array(box).astype(np.int32)
        # poly = poly.reshape((-1, 1, 2))
        # cv2.polylines(img, [poly], True, color=(0, 0, 255), thickness=2)
        x1 = box[0][0]
        x2 = box[2][0]
        y1 = box[0][1]
        y2 = box[2][1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(res_img_file, img)

