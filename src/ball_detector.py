import cv2
import vision_util as vision_common


def find_balls(gray):
    canny = vision_common.canny(gray, 50)
    cv2.imshow('canny', canny)

img = cv2.imread('/Users/vmagro/Desktop/balls.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
find_balls(gray)

cv2.waitKey(0)
