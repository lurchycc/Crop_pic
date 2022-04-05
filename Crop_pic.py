import cv2
from skimage.measure import label
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
def binary_mask_to_box(binary_mask):
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    # 取最大面积的连通区域
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bounding_box = [x, y, x + w, y + h]
    return bounding_box
#读取图片,RGB模式
img = cv2.imread('/media/lz/lz2/coco/train2017/003d60244b7d11e9a944305a3a77b88e.jpg',1)
#转化为灰度图
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img_gray,(5,5),0)
#阈值分割
ret,img_th_gaussian = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
bounding_box = binary_mask_to_box(img_th_gaussian)
minr, minc, maxr, maxc = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
ptLeftTop = (minr, minc)
ptRightBottom = (maxr, maxc)
point_color = (0, 0, 255)
thickness = 5
lineType = 4
cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
cv2.imwrite('res.jpg', img)
