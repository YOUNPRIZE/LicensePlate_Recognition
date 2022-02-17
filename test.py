from configparser import MAX_INTERPOLATION_DEPTH
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms
from craft import new
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import cv2 as cv
#from google.colab.patches import cv2_imshow
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#plt.plot(range(10))
#plt.show()

ocr_roi_path     = os.path.join('.', 'awb', 'roi')
ocr_image_Path   = os.path.join('.', 'awb', 'image')
ocr_contour_path = os.path.join('.', 'awb', 'contour')
ocr_data_path    = os.path.join('.', 'awb', 'data')

def awbImageRead(imageName) :
    start_time    = time.time()
    
    # Read Input Image
    imageFileName = f'{imageName}.jpg'
    img = cv.imread(imageFileName)
    height, width, channel = img.shape
    #cv2_imshow(img)

    # Convert Image to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Path settings
    image_roi_path = os.path.join(ocr_roi_path, imageName)
    if not os.path.isdir(image_roi_path):
        os.mkdir(image_roi_path)
    img = cv.imread(os.path.join(ocr_image_Path,imageFileName))
    plt.figure(figsize=(12,10))
    plt.imshow(gray, cmap='gray')
    img_result, roi_boxes, padded_image = new.getRoiBoxes(os.path.join(ocr_image_Path,imageFileName))
    #print(roi_boxes)

    # Make red boxes
    gray2 = cv.cvtColor(img_result, cv.COLOR_BGR2GRAY)
    plt.imshow(gray2, cmap='gray')
    img_out = img_result.copy()
    cv.rectangle(img_out, (232, 276), (284, 316), color=(255, 0, 0), thickness = 2)
    cv.rectangle(img_out, (328, 276), (420, 316), color=(255, 0, 0), thickness = 2)
    cv.rectangle(img_out, (276, 276), (328, 316), color=(255, 0, 0), thickness = 2)
    plt.figure(figsize=(12,10))
    plt.imshow(img_out)
    
    '''
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0, 255, -1)
    new_image = cv.bitwise_and(img, img, mask=mask)
    plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
    '''
    #print('Elapsed(Get ROI)  : ', time.time() - start_time)

################################################################################################

imageName = '1'
awbImageRead(imageName)