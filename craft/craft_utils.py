'''  
Copyright (c) 2019-present NAVER Corp.
MIT License
'''

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    
    ''' labeling method '''
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0) # threshold를 경계로 0, 1로 처리
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0) # threshold를 경계로 0, 1로 처리
    
    # text 배열과 link 배열 더한 후 clip을 이용해 최소 0, 최대 1로 처리
    # text 배열과 link 배열이 모두 1인 경우 더하면 2가 되지만 1로 처리하였음
    text_score_comb = np.clip(text_score + link_score, 0, 1) 

    # img = (np.clip(text_score_comb, 0, 1) * 255).astype(np.uint8)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    '''
        area     : cv2.CC_STAT_AREA]
        center_x : int(centroids[i, 0])
        center_y : int(centroids[i, 1]) 
        left     : cv2.CC_STAT_LEFT
        top      : cv2.CC_STAT_TOP
        width    : cv2.CC_STAT_WIDTH
        height   : cv2.CC_STAT_HEIGHT
    '''

    boxes = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        boxes.append(box)
        mapper.append(k)

    # return boxes, labels, mapper
    return boxes

def adjustResultCoordinates(boxes, ratio_w, ratio_h, ratio_net=2):
    if len(boxes) > 0:
        boxes = np.array(boxes)
        for k in range(len(boxes)):
            if boxes[k] is not None:
                boxes[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return boxes
