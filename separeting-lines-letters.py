import cv2
import numpy as np
from skimage import feature
import imutils
import matplotlib.pyplot as plt

img_orj = cv2.imread("stajBook5.jpeg")
img_orj = cv2.resize(img_orj, (1500, 1000))
gray = cv2.cvtColor(img_orj, cv2.COLOR_BGR2GRAY)
img_lines = img_orj.copy()
img_letters = img_orj.copy()
blur_gaus = cv2.GaussianBlur(gray,(3,3),0)
kernel = np.ones((3,3), np.uint8)

thresh_adaptive = cv2.adaptiveThreshold(blur_gaus,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,11)


def find_lines(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,10))
    morph_lines = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morph_lines.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_lines, (x, y), (x + w, y + h), (0, 255, 0), 1)
        area = cv2.contourArea(c)
        print("area  "+str(area))
        #cv2.drawContours(img_lines, [c], -1, (0, 0, 255), 2)


def find_letters(img):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph_letters = cv2.morphologyEx(img, cv2.MORPH_CROSS, rect_kernel)

    cnts = cv2.findContours(morph_letters.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    for (i, c) in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > 40:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img_letters, (x, y), (x + w, y + h), (0, 255, 0), 1)
            area = cv2.contourArea(c)
            print("area  "+str(area))
            #cv2.drawContours(img_letters, [c], -1, (0, 0, 255), 2)
    
    

line_img = find_lines(thresh_adaptive)
letter_img = find_letters(thresh_adaptive)

cv2.imshow("LINES", img_lines)
cv2.imshow("LETTERS", img_letters)
cv2.waitKey(0)
cv2.destroyAllWindows()
