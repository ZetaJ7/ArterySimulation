import math
import numpy as np
import cv2


def img_custom(img):
    # Resolution change to h=w for GUI
    h, w = np.shape(img)
    if h == w:
        img_new = img;
    elif h < w:
        delta = int(0.5 * (w - h))
        img_delta1 = np.zeros((delta, w), dtype=np.uint8)
        img_delta2 = np.zeros((w - h - delta, w), dtype=np.uint8)
        img_new = np.concatenate((img_delta1, img, img_delta2), axis=0)
    elif h > w:
        delta = 0.5 * (h - w).cast(int)
        img_delta1 = np.zeros((h, delta), dtype=np.uint8)
        img_delta2 = np.zeros((h, h - w - delta), dtype=np.uint8)
        img_new = np.concatenate((img_delta1, img, img_delta2), axis=1)
    img_new.astype(np.uint8)
    return img_new


# Canny函数获取边界
# def get_edge(img):
#     edge=cv2.Canny(img,0,50)
#     return edge

# 外扩边界
def get_edge(background):
    edge = np.zeros_like(background, dtype=int)
    h, w = np.shape(edge)
    for i in [0, h - 1]:
        for j in range(1, w - 1):
            if background[i][j] == 1:
                if background[i][j + 1] == 0:
                    edge[i][j + 1] = 255
                if background[i][j - 1] == 0:
                    edge[i][j - 1] = 255
    for j in [0, w - 1]:
        for i in range(1, h - 1):
            if background[i][j] == 1:
                if background[i - 1][j] == 0:
                    edge[i - 1][j] = 255
                if background[i + 1][j] == 0:
                    edge[i + 1][j] = 255

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if background[i][j] == 1:
                if background[i + 1][j] == 0:
                    edge[i + 1][j] = 255
                if background[i - 1][j] == 0:
                    edge[i - 1][j] = 255
                if background[i][j + 1] == 0:
                    edge[i][j + 1] = 255
                if background[i][j - 1] == 0:
                    edge[i][j - 1] = 255
    return edge


# # 非外扩边界(可能会封闭窄流道区域)
# def get_edge(background):
#     edge = np.zeros_like(background,dtype=int)
#     h,w=np.shape(edge)
#     for i in [0,h-1]:
#         for j in range (1,w-1):
#             if background[i][j]==50:
#                 if background[i][j+1]==0:
#                     edge[i][j]=255
#                 if background[i][j-1]==0:
#                     edge[i][j]=255
#     for j in [0,w-1]:
#         for i in range (1,h-1):
#             if background[i][j]==50:
#                 if background[i-1][j]==0:
#                     edge[i][j]=255
#                 if background[i+1][j]==0:
#                     edge[i][j]=255
#
#     for i in range(1,h-1):
#         for j in range(1,w-1):
#             if background[i][j]==50:
#                 if background[i+1][j]==0:
#                     edge[i][j]=255
#                 if background[i-1][j]==0:
#                     edge[i][j]=255
#                 if background[i][j+1]==0:
#                     edge[i][j]=255
#                 if background[i][j-1]==0:
#                     edge[i][j]=255
#     return edge

def get_angle(img, edge):
    # Sobel should deal with format "uint8"(0-255)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    angle = np.zeros_like(edge, dtype=float)
    for i in range(np.size(edge, 0)):
        for j in range(np.size(edge, 1)):
            if edge[i][j] != 0:
                # angle[i][j] = math.atan2(abs(sobel_x[i][j]), abs(sobel_y[i][j]))
                angle[i][j] = math.atan2(sobel_x[i][j], sobel_y[i][j])  # 弧度单位
                # angle[i][j] = math.atan2(abs(sobel_x[i][j]), abs(sobel_y[i][j]))*180/math.pi   # 角度单位
    return angle


def get_param(img1, entrance='left', exit='right'):
    # return
    # h:height of the image
    # w:width of the image
    # background : the BC and fluid area of the image
    #              "1" for fluid area, "0" for BC and wall area，“255” for edge
    # edge: the edge of artery ,value "255"
    # angle: the angle of each point on edge
    img = img_custom(img1)
    crt_val = 100  # Critical value
    h = np.size(img, 0)
    w = np.size(img, 1)
    # Transform img to a (0,1) numpy
    background = np.zeros_like(img, dtype=int)
    for i in range(h):
        for j in range(w):
            if img[i][j] >= crt_val:
                background[i, j] = 1
    # get edge by Canny
    background1 = background.astype(np.uint8)
    edge = get_edge(background)
    for i in range(h):
        for j in range(w):
            if edge[i][j] == 255:
                background[i, j] = 255
    # get angle by gradient
    angle = get_angle(img, edge)  # Use img instead of background for more realistic angle
    bc_back = bc_area(background, entrance=entrance, exit=exit)
    entrance_min, entrance_max, entrance_mid = reset_area(bc_back, entrance=entrance)
    exit_mid = exit_area(bc_back, exit=exit)
    return h, w, background, edge, angle, bc_back, entrance_min, entrance_max, entrance_mid, exit_mid


def bc_area(background, entrance, exit):
    if entrance not in ["left", "up", "right", "down"]:
        raise ValueError("Entrance setups not in [left,up,right,down] please check!")
    if exit not in ["left", "up", "right", "down"]:
        raise ValueError("Exit setups not in  [left,up,right,down], please check!")
    if entrance == exit:
        raise AssertionError('Entrance and Exit on Same side is not supported yet')
    # bc_back: Record entrance and exit area of artery, "1" for entrance and "2" for exit
    bc_back = np.zeros_like(background)
    h, w = background.shape[0], background.shape[1]
    # Entrance area (flag=1)
    if entrance == 'up':
        for i in range(5):
            for j in range(w):
                if background[i][j] == 1:
                    bc_back[i][j] = 1
    if entrance == 'down':
        for i in range(h - 5, h):
            for j in range(w):
                if background[i][j] == 1:
                    bc_back[i][j] = 1
    if entrance == 'left':
        for i in range(h):
            for j in range(5):
                if background[i][j] == 1:
                    bc_back[i][j] = 1
    if entrance == 'right':
        for i in range(h):
            for j in range(w - 5, w):
                if background[i][j] == 1:
                    bc_back[i][j] = 1
    # Exit area (flag=2)
    if exit == 'up':
        for i in range(1):
            for j in range(w):
                if background[i][j] == 1:
                    bc_back[i][j] = 2
    if exit == 'down':
        for i in range(h - 1, h):
            for j in range(w):
                if background[i][j] == 1:
                    bc_back[i][j] = 2
    if exit == 'left':
        for i in range(h):
            for j in range(1):
                if background[i][j] == 1:
                    bc_back[i][j] = 2
    if exit == 'right':
        for i in range(h):
            for j in range(w - 1, w):
                if background[i][j] == 1:
                    bc_back[i][j] = 2

    return bc_back


def reset_area(bc_back, entrance='left'):
    h = np.size(bc_back, 0)
    w = np.size(bc_back, 1)
    label_min = []
    label_max = []
    mid_label = []
    if entrance == 'left':
        for j in range(h - 1):
            if bc_back[j][0] == 0 and bc_back[j + 1][0] == 1:
                label_min.append(j + 1)
            if bc_back[j][0] == 1 and bc_back[j + 1][0] == 0:
                label_max.append(j)
    if entrance == 'right':
        for j in range(h - 1):
            if bc_back[j][w-1] == 0 and bc_back[j + 1][w-1] == 1:
                label_min.append(j + 1)
            if bc_back[j][w-1] == 1 and bc_back[j + 1][w-1] == 0:
                label_max.append(j)
    if entrance == 'up':
        for i in range(w - 1):
            if bc_back[0][i] == 0 and bc_back[0][i + 1] == 1:
                label_min.append(i + 1)
            if bc_back[0][i] == 1 and bc_back[0][i + 1] == 0:
                label_max.append(i)
    if entrance == 'down':
        for i in range(w - 1):
            if bc_back[h-1][i] == 0 and bc_back[h-1][i + 1] == 1:
                label_min.append(i + 1)
            if bc_back[h-1][i] == 1 and bc_back[h-1][i + 1] == 0:
                label_max.append(i)

    # Mid label count and calculate
    if len(label_max) == 0 and len(label_min) == 0:
        raise AssertionError('Reset area is empty, please check!')
    if len(label_max) == len(label_min):
        if label_min[0] < label_max[0]:
            count = len(label_max)
            for i in range(count):
                mid_label.append(0.5 * (label_max[i] + label_min[i]))
        if label_min[0] > label_max[0]:
            count = len(label_max) + 1
            mid_label.append(0.5 * label_max[0])
            for i in range(count - 2):
                mid_label.append(0.5 * (label_max[i + 1] + label_min[i]))
            mid_label.append(0.5 * (label_min[count - 2] + w))

    if len(label_max) > len(label_min):
        count = len(label_max)
        mid_label.append(0.5 * label_max[0])
        for i in range(count - 1):
            mid_label.append(0.5 * (label_max[i + 1] + label_min[i]))

    if len(label_max) < len(label_min):
        count = len(label_min)
        for i in range(count - 1):
            mid_label.append(0.5 * (label_max[i] + label_min[i]))
        mid_label.append(0.5 * (label_min[count - 1] + w))

    return label_min, label_max, mid_label


def exit_area(bc_back, exit = 'right'):
    h = np.size(bc_back, 0)
    w = np.size(bc_back, 1)
    label_min = []
    label_max = []
    mid_label = []
    if exit == 'left':
        for j in range(h - 1):
            if bc_back[j][0] == 0 and bc_back[j + 1][0] == 2:
                label_min.append(j + 1)
            if bc_back[j][0] == 2 and bc_back[j + 1][0] == 0:
                label_max.append(j)
    if exit == 'right':
        for j in range(h - 1):
            if bc_back[j][w-1] == 0 and bc_back[j + 1][w-1] == 2:
                label_min.append(j + 1)
            if bc_back[j][w-1] == 2 and bc_back[j + 1][w-1] == 0:
                label_max.append(j)
    if exit == 'up':
        for i in range(w - 1):
            if bc_back[0][i] == 0 and bc_back[0][i + 1] == 2:
                label_min.append(i + 1)
            if bc_back[0][i] == 2 and bc_back[0][i + 1] == 0:
                label_max.append(i)
    if exit == 'down':
        for i in range(w - 1):
            if bc_back[h-1][i] == 0 and bc_back[h-1][i + 1] == 2:
                label_min.append(i + 1)
            if bc_back[h-1][i] == 2 and bc_back[h-1][i + 1] == 0:
                label_max.append(i)

    # Mid label count and calculate
    if len(label_max) == 0 and len(label_min) == 0:
        raise AssertionError('Exit area is empty, please check!')

    if len(label_max) == len(label_min):
        if label_min[0] < label_max[0]:
            count = len(label_max)
            for i in range(count):
                mid_label.append(0.5 * (label_max[i] + label_min[i]))
        if label_min[0] > label_max[0]:
            count = len(label_max) + 1
            mid_label.append(0.5 * label_max[0])
            for i in range(count - 2):
                mid_label.append(0.5 * (label_max[i + 1] + label_min[i]))
            mid_label.append(0.5 * (label_min[count - 2] + w))

    if len(label_max) > len(label_min):
        count = len(label_max)
        mid_label.append(0.5 * label_max[0])
        for i in range(count - 1):
            mid_label.append(0.5 * (label_max[i + 1] + label_min[i]))

    if len(label_max) < len(label_min):
        count = len(label_min)
        for i in range(count - 1):
            mid_label.append(0.5 * (label_max[i] + label_min[i]))
        mid_label.append(0.5 * (label_min[count - 1] + w))

    # return label_min, label_max, mid_label
    return mid_label

# test
filepath = 'imgs/artery5.png'
entrance='up'
exit='down'
img = cv2.imread(filepath, 0)
height, width, background, edge, angle, bc_back, entrance_min, entrance_max, entrance_mid, exit_mid  = get_param(img, entrance=entrance, exit=exit)
# a=bc_back[64][0]
# print(a)
print(exit_mid)
print(entrance_mid)
print('Test finished!')
