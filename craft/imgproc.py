'''  
Copyright (c) 2019-present NAVER Corp.
MIT License
'''

import numpy as np
from skimage import io
import cv2

def loadImage(img_file):
    img = io.imread(img_file) # RGB order
    if img.shape[0]   == 2 : img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2]   == 4 : img = img[:,:,:3]
    img = np.array(img)
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, target_width, image_down_ratio=4):
    # img = resize_margin_off(img)
    image_height, image_width, channel = img.shape
    
    resize_width  = round(target_width)
    resize_ratio  = resize_width / image_width
    resize_height = round(resize_ratio * image_height)
    resize_image = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    resize_ratio  = 1/image_down_ratio
    resize_width  = round(resize_width  * resize_ratio) 
    resize_height = round(resize_height * resize_ratio)

    # make canvas and paste image
    # vgg에 대입하기 위해서는 이미지 size가 32의 배수 이여야 함
    # 따라서 32로 나눠서 나머지가 발생한다면 32에 맞춰서 0을 채운 배열(빈캔버스)를 만들고
    # 실제이미지를 그 캔버스에 대입한다.(대입은 왼쪽 모서리부터 채운다.)
    square_height, square_width = resize_height, resize_width
    if resize_height % 32 != 0:
        square_height = resize_height + (32 - resize_height % 32)
    if resize_width % 32 != 0:
        square_width = resize_width + (32 - resize_width % 32)
    padded_image = np.zeros((square_height, square_width, channel), dtype=np.float32)
    padded_image[0:resize_height, 0:resize_width, :] = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    return resize_image, padded_image, resize_ratio

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def resize_margin_off(img) :
    dst = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 2000, 2000, apertureSize = 5, L2gradient = True)
    lines = cv2.HoughLinesP(canny, 0.5, np.pi / 180, 90, minLineLength = 800, maxLineGap = 400)
    
    min_x1 = 350
    min_y1 = 350
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        if w < 0 or h > 1800 :
            if x1 > 150 :                
                min_x1 = (min_x1 > x1) and x1 or min_x1
                cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if w > 1000 or h < 0 :
            if y1 > 150 and y1 < 500 :
                min_y1 = (min_y1 > y1) and y1 or min_y1
                cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # ''' 가로줄의 왼쪽 끝이 150보다 떨어져서 시작하면 가로줄의 최저점을 B/L의 시작접으로 처리한다. '''
                if x1 > 350 :
                    min_x1 = (min_x1 > x1) and x1 or min_x1
        
    image_height, image_width, channel = img.shape    
    x1 = min_x1 - 100
    x2 = x1 + 2200 
    y1 = min_y1 - 100
    y2 = y1 + 2800
    return img[y1:y2, x1:x2]
    
