from picamera import PiCamera
import picamera.array
import time
import cv2
import numpy as np

def get_now_cord(srcedge,templateedge):
    res=cv2.matchTemplate(srcedge.copy(),templateedge,cv2.TM_CCOEFF_NORMED)
    _,_,_,max_loc=cv2.minMaxLoc(res)
    top_left=max_loc
    return(top_left[0]+15,top_left[1]+58)

img = cv2.imread('game_zone.jpg',1)
templateedge=cv2.imread('templateedge.jpg',0)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,150,250,apertureSize=5)

now_cord=get_now_cord(edge,templateedge)

img_cut=img[100:now_cord[1],10:-10]
img_cut=cv2.GaussianBlur(img_cut,(5,5),0)
cv2.imwrite('img_cut.jpg',img_cut)
gray=cv2.cvtColor(img_cut,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,50,250,apertureSize=5)
cv2.imwrite('cacacaca.jpg',edge)
"""

cv2.imwrite('imgASGASGSAGSGA_cut.jpg',gray)
thth=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imwrite('eheh.jpg',thth)
"""





"""
cv2.imwrite('img_cut.jpg',img_cut)

cv2.imwrite('img_cutslice.jpg',img_cut[:10,:])

color_slice_hsv=img_cut[:10,:]
print(color_slice_hsv)
print(np.mean(np.mean(color_slice_hsv,axis=0),axis=0))

lower=np.array([200,200,200])
upper=np.array([240,240,240])
mask=cv2.inRange(img_cut,lower,upper)
mask=np.abs(mask-255)
cv2.imwrite('mask.jpg',cv2.bitwise_and(img_cut,img_cut,mask=mask))
"""