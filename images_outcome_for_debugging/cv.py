from picamera import PiCamera
import picamera.array
import time
import cv2
import numpy as np
def finmaxcontour(src):   
    _,contours,hierarchy=cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy=np.squeeze(hierarchy)
    #print(hierarchy)
    chosen_index=0
    max_area=0
    index=0
    #print(type(index))
    while index !=-1:
        area_tem=cv2.contourArea(contours[index])
        if area_tem>max_area:
            max_area=area_tem
            chosen_index=index
        index=hierarchy[index][0]
    cnt=contours[chosen_index]
    return contours,hierarchy,chosen_index,cnt

img=cv2.imread('test3.jpg',1)
#hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('image.jpg',img)
cv2.imwrite('gray.jpg',gray)

corners=cv2.goodFeaturesToTrack(gray,20,0.01,50)
corners=np.int0(corners)

for i in corners:
    x,y=i.ravel()
    cv2.circle(gray,(x,y),3,255,-1)
cv2.imwrite('last.jpg',gray)

_,thresh=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
#thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                       cv2.THRESH_BINARY,11,2)
#print(hsv)
#print(gray)
#print(thresh)
cv2.imwrite('image.jpg',img)


contours,hierarchy,chosen_index,cnt=finmaxcontour(thresh)
#cnt=cnt.reshape((-1,1,2))
#cv2.polylines(thresh,[cnt],True,(0,255,0))
print(thresh)
cv2.fillPoly(thresh,[cnt],(255,255,255))
print(thresh)
cv2.imwrite('th.jpg',thresh)
thresh=np.float32(thresh)
dst=cv2.cornerHarris(thresh,9,29,0.04)
print(dst)
print(type(dst))
dst=cv2.dilate(dst,None)

img[dst>0.6*dst.max()]=[0,0,255]
cv2.imwrite('harris.jpg',img)


while 1:
    pass







"""
kernel=np.ones((19,19))
thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
cv2.imwrite('thfill.jpg',thresh)

contours,hierarchy,chosen_index,cnt=finmaxcontour(thresh)

approx=cv2.approxPolyDP(cnt,0.11*cv2.arcLength(cnt,True),True)
#print(approx)
app_img=img.copy()
cv2.drawContours(app_img,[approx],0,(0,255,0),3)
cv2.imwrite('app_img.jpg',app_img)
"""

"""
corner------dect

line  -- corner

fit an poly  wrong

x+y??

shape matching ??
"""


rect=cv2.minAreaRect(cnt)
box=cv2.boxPoints(rect)
box=np.int0(box)
box_img=np.copy(img)

cv2.drawContours(box_img,[box],0,(0,0,255),3)
cv2.imwrite('box.jpg',box_img)
cnt_img=np.copy(img)
cv2.drawContours(cnt_img,contours,chosen_index,(0,255,0),3)
cv2.imwrite('cnt.jpg',cnt_img)

"""
cam=PiCamera()
for i in range(5):
    cam.start_preview()
    time.sleep(2)
    cam.capture('test'+str(i)+'.jpg')
    cam.stop_preview()
    time.sleep(2)
"""
    