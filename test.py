import cv2
import numpy as np

image = cv2.imread('5.jpg')
result = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 11)
#cv2.imshow("thresh",thresh)
# Fill rectangular contours
# CHECK OTHER CONTOUR SETTINGS ? TO EXLCUDE OUTER ?
# https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
# https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
#cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)
    cv2.drawContours(thresh, [c], -1, (0, 0, 0), 1)

# Morph open
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 6))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
#cv2.imshow("opening",opening)
# Draw rectangles
#cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(image.shape[:3], dtype=np.uint8)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.drawContours(mask, [c], -1, (255,255,255), -1) # fill conours
    #                 #cv2.bitwise_and(image, image, mask=mask)
 
 #detect defected part in mask   

rm_BG = cv2.bitwise_and(image, mask)
imgray = cv2.cvtColor(rm_BG, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 107, 200, 0, cv2.THRESH_BINARY)
#edged = cv2.Canny(imgray, 110, 255,apertureSize=3)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (255,0,0), 2)

count = 0 
# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][2] == -1:
        # if the size of the contour is greater than a threshold  
            if cv2.contourArea(cnt) > 55 :
                x,y,w,h = cv2.boundingRect(cnt)
                if (w>1 and w<50) and (h>1 and h<50):
                    cv2.drawContours(rm_BG, [cnt], -1, (0,255,0), -1)

    

#print(image.shape,mask.shape)
# rm_BG = cv2.bitwise_and(image, mask)
cv2.imshow('mask', mask)
cv2.imshow('img', image)
cv2.imshow('rm_BG', rm_BG)
cv2.waitKey()
