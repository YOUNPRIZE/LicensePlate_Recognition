'''
Copyright (c) 2020-present HFOR Corp.
'''

import argparse
import os
import time
import numpy as np
import pandas as pd
from   collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from   torch.autograd import Variable
from matplotlib import pyplot as plt

from . import craft_utils
from . import file_utils
from . import imgproc
from .craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def logis_roi_net(net, image, text_threshold=0.1, link_threshold=0.4, low_text=0.3):    
    
    # 너비 설정
    target_width = 640
    
    # 얼마나 줄일건지?
    image_down_ratio = 2
    
    # resize된 이미지를 matplot으로 찍어보기
    resize_image, padded_image, resize_ratio = imgproc.resize_aspect_ratio(image, target_width=target_width, image_down_ratio=image_down_ratio)
    #print("출력")
    #cv2_imshow(resize_image)
    #cv2_
    ratio_h = ratio_w = 1 / resize_ratio

    # 네트워크에 대입하기 위해 (채널, 세로, 가로)로 변경
    x = imgproc.normalizeMeanVariance(padded_image)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    
    # 네트워크 forward 실행
    with torch.no_grad():
        y, feature = net(x)

    # 예측값을 text, link로 구분함
    score = y.cpu().data.numpy()
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # 예측값을 Threshold를 이용해서 유효한 ROI Box만 필터링
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)

    # Box의 좌표를 원본 이미지의 좌표로 환산.
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return resize_image, boxes

def getRoiBoxes(image_path):
    net = CRAFT(pretrained=False)  # initialize
    net.load_state_dict(copyStateDict(torch.load('./weights/hurap_ocr_logis.pth', map_location='cpu')))
    net.eval()
    
    image = imgproc.loadImage(image_path) # 이미지를 rgb로 읽어서 np array로 반환함
    
    # Certainity required for a something to be classified as a letter. 
    # The higher this value the clearer characters need to look. I'd recommend 0.5-0.6
    # 값이 높을수록 문자가 더 깔끔해야 함(흐릿하면 ㄴㄴ).
    text_threshold = 0.6
    
    # Amount of distance allowed between two characters for them to be seen as a single word. 
    # I recommend 0.1-0.5, however playing with this value for your own use case might be better
    # 두 개의 문자를 거리에 따라 한 개의 문자로 볼 것인지?
    link_threshold = 1
    
    # Amount of boundary space around the letter/word when the coordinates are returned. 
    # The higher this value the less space. 
    # Upping this value also affects the link threshold of seeing words as one, but it can cut off unecessary borders around leters. 
    # Having this value too high can affect edges of letters, cutting them off and lowering accuracy in reading them. 
    # I'd recommend 0.3-0.4
    # 값이 클수록 여백이 없음 -> recognition 하기가 어려워짐
    low_text = 0.4
    resize_image, boxes = logis_roi_net(net, image, text_threshold=text_threshold, link_threshold=link_threshold, low_text=low_text)
    
    #print(boxes[0][0])
    #print(type(boxes[0][0]))
    # resize_image, boxes = logis_roi_net(net, image)

    # boxes를 dataframe으로 처리
    row = {'x1': 0,'y1': 0,'x2': 0, 'y2': 0, 'w': 0, 'h': 0}
    roi_boxes = pd.DataFrame(columns=row.keys())
    for i, box in enumerate(boxes):
        box = np.array(box).astype(np.int32)
        newRow = row.copy()
        newRow['x1'] = box[0][0] < box[3][0] and box[0][0] or box[3][0]
        newRow['x2'] = box[1][0] > box[2][0] and box[1][0] or box[2][0]
        newRow['y1'] = box[0][1] < box[1][1] and box[0][1] or box[1][1]
        newRow['y2'] = box[2][1] > box[3][1] and box[2][1] or box[3][1]
        roi_boxes = roi_boxes.append(newRow, ignore_index=True)
    
    return resize_image, roi_boxes
    
    #return resize_image, boxes, padded_image