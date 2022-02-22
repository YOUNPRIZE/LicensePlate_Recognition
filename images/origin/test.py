from configparser import MAX_INTERPOLATION_DEPTH
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms
import sys
import os
# 현재 결대경로는 carplate
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from craft import new
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#plt.plot(range(10))
#plt.show()
import easyocr

# 현재 절대경로가 carplate 기준이므로 절대경로에 따른 상대경로 설정
origin_path = os.path.join('images', 'origin', 'img')
contour_path = os.path.join('images', 'contour')
rectangle_path = os.path.join('images', 'rectangle')
#roi_path = os.path.join('images', 'roi')

def ImageRead(imageName):
    # Read Input Image
    # 해당 이미지 Read
    
    imageFileName = f'{imageName}.jpg'
    img = cv.imread(os.path.join(origin_path, imageFileName))

    # Search roi_boxes in the image
    # Resized Image 와 ROI boxes 추출
    img_result, roi_boxes = new.getRoiBoxes(os.path.join(origin_path, imageFileName))
    
    # Save cropped roi_box images
    # ROI boxes 부분만 crop해서 추출 후 저장
    for i in range(len(roi_boxes)):
        img_crop = img_result[roi_boxes.loc[i,'y1']:roi_boxes.loc[i,'y2'], roi_boxes.loc[i,'x1']:roi_boxes.loc[i,'x2']]
        new_roi_path = os.path.join(rectangle_path, imageName)
        if not os.path.isdir(new_roi_path):
            os.mkdir(new_roi_path)
        cv.imwrite(os.path.join(new_roi_path, f'{i}.jpg'), img_crop)

    # 이 곳에 roi_boxes들 중에서 번호판만 나타내는 boxes만 찾아서 뽑아내야 함.
    
    # Read the text using easyocr
    # easyocr의 recognition 기능 사용해서 숫자 읽기
    cropped_image = img_result[roi_boxes['y1'].min():roi_boxes['y2'].max(), roi_boxes['x1'].min():roi_boxes['x2'].max()]
    reader = easyocr.Reader(["ko"], gpu=False)
    result = reader.recognize(cropped_image)
    fourNumb = result[0][1][-4:]
    print(result)
    
    # Save the rectangled image 
    # 번호판 부분 Rectangle 처리, 숫자 삽입 후 이미지 저장
    img_rect = img_result.copy() 
    cv.rectangle(img_rect, (roi_boxes['x1'].min(), roi_boxes['y1'].min()), (roi_boxes['x2'].max(), roi_boxes['y2'].max()), color=(0, 255, 0), thickness = 2)
    point = roi_boxes['x1'].min(), roi_boxes['y1'].min() - 5
    cv.putText(img_rect, f'{fourNumb}', point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA, bottomLeftOrigin=False)
    cv.imwrite(os.path.join(rectangle_path, f'{imageName}.jpg'), img_rect)
    
################################################################################################

if __name__ == "__main__":
    imageName = '20'
    start_time    = time.time()
    
    '''
    file_list = os.listdir(origin_path)
    file_name = []
    for file in file_list:
        if file.count(".") == 1: 
            name = file.split('.')[0]
            file_name.append(name)
        else:
            for k in range(len(file)-1,0,-1):
                if file[k]=='.':
                    file_name.append(file[:k])
                    break
    for i in file_name:
        ImageRead(i)
    '''
    
    ImageRead(imageName)
    print("%s 초" % (time.time() - start_time))    