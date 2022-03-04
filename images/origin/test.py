from configparser import MAX_INTERPOLATION_DEPTH
from typing import Type
#from socket import send_fds
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms
import sys
import os
import re
# 현재 결대경로는 carplate
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from craft import new
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import easyocr

# 현재 절대경로가 carplate 기준이므로 절대경로에 따른 상대경로 설정
origin_path = os.path.join('images', 'origin', 'img')
contour_path = os.path.join('images', 'contour')
rectangle_path = os.path.join('images', 'rectangle')
sharpen_path = os.path.join('images', 'sharpen')
unrecognized_path = os.path.join('images', 'unrecognized')
#roi_path = os.path.join('images', 'roi')

# openCV imwrite 사용 시 한글로 된 경로가 포함되어 있으면 에러 발생
# 따라서 해당 함수 사용
def Imagewrite(img, imageName, path):
    extension = os.path.splitext(f'{imageName}.jpg')[1]
    result, encoded_img = cv.imencode(extension, img)
    if result:
        with open(os.path.join(path, f'{imageName}.jpg'), mode='w+b') as f:
            encoded_img.tofile(f)

# 특수문자가 포함된 문자열을 공백으로 대체해주는 함수
def clean_text(inputString):
        text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', inputString)
        return text_rmv

def imageDenoise(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.fastNlMeansDenoising(img, h = 10, templateWindowSize=7, searchWindowSize=21)
    gaussian = cv.GaussianBlur(img, (0,0), 3)
    sharpen = cv.addWeighted(img, 1.5, gaussian, -0.5, 0)
    binarized = cv.adaptiveThreshold(sharpen, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 15)
    return gaussian, sharpen, binarized
    
def ImageRead(imageName):
    #start_time    = time.time()
    print(imageName)
    
    # 1. Read Input Image
    # 해당 이미지 Read
    imageFileName = f'{imageName}.jpg'
    
    ## 경로에 한글 포함돼있으면 imread 불가능하므로 아래와 같은 방법으로 image 불러오기
    img_array = np.fromfile(os.path.join(origin_path, imageFileName), np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    ## 이미지 잡음 제거
    _, sharpen, _ = imageDenoise(img)
    sharpen = cv.medianBlur(sharpen, 3)
    ## 잡음 제거된 이미지 저장
    Imagewrite(sharpen, imageName, sharpen_path)
    #img = cv.imread(os.path.join(origin_path, imageFileName))
    
    # 2. Search roi_boxes in the image
    ## Resized Image 와 ROI boxes 추출
    img_result, roi_boxes = new.getRoiBoxes(os.path.join(sharpen_path, imageFileName))

    if len(roi_boxes) == 0:
        global cannotRecog
        cannotRecog += 1
        Ineed = 0
        fourNumb = 0
        return Ineed, fourNumb, img_result
        
    # 3. Save cropped roi_box images
    ## 원래 이미지에 ROI boxes 표시
    img_contour = img_result.copy()
    for i in range(len(roi_boxes)):
        cv.rectangle(img_contour, (roi_boxes.loc[i,'x1'], roi_boxes.loc[i,'y1']), (roi_boxes.loc[i,'x2'], roi_boxes.loc[i,'y2']), color=(0, 255, 0), thickness = 2)
    Imagewrite(img_contour, imageName, contour_path)
    
    ## ROI boxes 개별적으로 crop해서 추출 후 저장
    for i in range(len(roi_boxes)):
        img_crop = img_result[roi_boxes.loc[i,'y1']:roi_boxes.loc[i,'y2'], roi_boxes.loc[i,'x1']:roi_boxes.loc[i,'x2']]
        result = reader.recognize(img_crop)
        #print(result[0][1])
        new_roi_path = os.path.join(rectangle_path, imageName)
        if not os.path.isdir(new_roi_path):
            os.mkdir(new_roi_path)
        Imagewrite(img_crop, i, new_roi_path)
    
    # 4. Read the text using easyocr
    ## easyocr의 recognition 기능 사용해서 숫자 읽기
    cropped_image = img_result[roi_boxes['y1'].min():roi_boxes['y2'].max(), roi_boxes['x1'].min():roi_boxes['x2'].max()]
    result = reader.recognize(cropped_image)
    
    if result[0][1].isalnum() == False:
        newResult = clean_text(result[0][1])
        Ineed = newResult.replace(" ", "")
        fourNumb = newResult[-4:]
    else:
        Ineed = result[0][1].replace(" ", "")
        fourNumb = result[0][1][-4:]
    #print(Ineed, fourNumb)
    
    # 5. Save the rectangled image 
    ## 번호판 부분 Rectangle 처리, 숫자 삽입 후 이미지 저장
    img_rect = img_result.copy() 
    cv.rectangle(img_rect, (roi_boxes['x1'].min(), roi_boxes['y1'].min()), (roi_boxes['x2'].max(), roi_boxes['y2'].max()), color=(0, 255, 0), thickness = 2)
    point = roi_boxes['x1'].min(), roi_boxes['y1'].min() - 5
    cv.putText(img_rect, f'{fourNumb}', point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA, bottomLeftOrigin=False)
    Imagewrite(img_rect, imageName, rectangle_path)
    
    #print("%s 초" % (time.time() - start_time))
    
    return Ineed, fourNumb, img_rect
    
################################################################################################

if __name__ == "__main__":    
    total_start_time    = time.time()
    
    # 폴더 안에 있는 이미지 파일들 이름 추출 후 리스트 생성
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

    cannotRecog = 0
    tot_recog = 0
    num_recog = 0
    
    # easyocr 모듈 호출
    reader = easyocr.Reader(["ko"], gpu=False)
    
    # 위에서 만든 리스트에 들어있는 파일 전부 ImageRead
    for i, v in enumerate(file_name):
        needed, real, imgUnrecog = ImageRead(v)
        cnt = i + 1
        # 전체 번호판
        if v[0:7] == needed:
            tot_recog += 1
        # 뒤의 4자리 번호   
        if v[3:7] == real:
            num_recog += 1
        else:
            cannotRecog += 1
            Imagewrite(imgUnrecog, v, unrecognized_path)
        print(f"{cnt} / {len(file_name)}")
        
    perc = (tot_recog / len(file_name)) * 100
    num_perc = (num_recog / len(file_name)) * 100
    totalTime = round(time.time() - total_start_time, 2)
    averageTime = totalTime / len(file_name)
    sentence = f"번호판 전체의 인식률은 {round(perc)} % 입니다."
    sentence2 = f"네 자리 숫자의 인식률은 {round(num_perc)} % 입니다."
    sentence3 = f"인식하지 못한 번호판은 총 {cannotRecog}개 입니다."
    sentence4 = f"한 이미지당 걸린 평균 시간은 {averageTime}초이고, 전체 걸린 시간은 {totalTime}초 입니다."
    print(sentence)
    print(sentence2)
    print(sentence3)
    print(sentence4)