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
#import easyocr
from craft import read

# 현재 절대경로가 carplate 기준이므로 절대경로에 따른 상대경로 설정
origin_path = os.path.join('images', 'origin', 'img')
contour_path = os.path.join('images', 'contour')
rectangle_path = os.path.join('images', 'rectangle')
#roi_path = os.path.join('images', 'roi')

def awbImageRead(imageName) :
    #start_time    = time.time()

    # Read Input Image
    imageFileName = f'{imageName}.jpg'
    print(imageFileName)
    img = cv.imread(os.path.join(origin_path, imageFileName))
    #height, width, channel = img.shape

    # Search roi_boxes in the image
    img_result, roi_boxes, padded_image = new.getRoiBoxes(os.path.join(origin_path, imageFileName))
    
    # Save cropped roi_box images
    for i in range(len(roi_boxes)):
        img_crop = img_result[roi_boxes.loc[i,'y1']:roi_boxes.loc[i,'y2'], roi_boxes.loc[i,'x1']:roi_boxes.loc[i,'x2']]
        new_roi_path = os.path.join(rectangle_path, imageName)
        if not os.path.isdir(new_roi_path):
            os.mkdir(new_roi_path)
        cv.imwrite(os.path.join(new_roi_path, f'{i}.jpg'), img_crop)
    
    # x1 값이 작은 순서대로 crop된 파일을 저장하고 그 x1 값을 가진 열의 index를 조회해서 그 열을 삭제하고 루프 돌리기 
    # 위 방법처럼 하려면 roi_boxes를 새로운 변수에 저장 후, while문 돌리기
    # 위의 방식인데 다시 생각해서 해보기. 해당 인덱스를 가진 행이 삭제되지 않아서 무한으로 루프가 실행됨.
    '''
    new_roi_boxes = roi_boxes.copy()
    numb = 0
    while new_roi_boxes.empty == False:
        index = new_roi_boxes.index[new_roi_boxes['x1'] == new_roi_boxes['x1'].min()].tolist()
        img_crop = img_result[roi_boxes.loc[index[0],'y1']:roi_boxes.loc[index[0],'y2'], roi_boxes.loc[index[0],'x1']:roi_boxes.loc[index[0],'x2']]
        new_roi_path = os.path.join(roi_path, imageName)
        if not os.path.isdir(new_roi_path):
            os.mkdir(new_roi_path)
        cv.imwrite(os.path.join(new_roi_path, f'{numb}.jpg'), img_crop)
        new_roi_boxes.drop(index[0])
        numb += 1
    '''
        
    # 이 곳에 roi_boxes들 중에서 번호판만 나타내는 boxes만 찾아서 뽑아내야 함.
    
    
    
    # Save the rectangled image 
    img_rect = img_result.copy() 
    cv.rectangle(img_rect, (roi_boxes['x1'].min(), roi_boxes['y1'].min()), (roi_boxes['x2'].max(), roi_boxes['y2'].max()), color=(255, 0, 0), thickness = 2)
    cv.imwrite(os.path.join(rectangle_path, f'{imageName}.jpg'), img_rect)
    
    # Read the text using easyocr
    cropped_image = img_result[roi_boxes['y1'].min():roi_boxes['y2'].max(), roi_boxes['x1'].min():roi_boxes['x2'].max()]
    print("xxxx")
    #reader = easyocr.Reader(["ko"], gpu=False)
    #print(reader)
    result = read.readtext(cropped_image)
    print(result)
    
    # Convert Image to Grayscale
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

################################################################################################


if __name__ == "__main__":
    imageName = '1'
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
        awbImageRead(i)
    '''
    
    awbImageRead(imageName)
    print("%s 초" % (time.time() - start_time))    
    
    