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


def draw_lines(src,lines):
    copy=src.copy()
    for line in lines:
        rho,theta=line
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+1000*(-b))
        y1=int(y0+1000*(a))
        x2=int(x0-1000*(-b))
        y2=int(y0-1000*(a))
        cv2.line(copy,(x1,y1),(x2,y2),255,1)
    cv2.imwrite('thresline.jpg',copy)
    return copy


def find_4cords(edge):
    lines=cv2.HoughLines(edge,1,np.pi/180,110)
    lines=np.squeeze(lines)
    lines[:,1]-=np.pi*(lines[:,0]<0)
    lines[:,0]=np.abs(lines[:,0])
    #print(lines)
    line_tem=np.zeros((4,2))
    """
    0:theta 1-2  p 500-900  
    1:theta 1-2  p 0-200
    2:theta -0.5 0.5 p300 500
    3:theta -0.5 0.5 p600 900
    """
    #print(lines)
    for line in lines:
        rho,theta=line
        if 1<theta and theta<2:
            if 500<rho and rho<900:
                line_tem[0]=[rho,theta]
            if 0<rho and rho<200:
                line_tem[1]=[rho,theta]
        if -0.5<theta and theta<0.5:
            if 300<rho and rho<500:
                line_tem[2]=[rho,theta]
            if 600<rho and rho<900:
                line_tem[3]=[rho,theta]
    lines=line_tem
    draw_lines(edge,lines)
    #print(lines)
    """
    0 zoushang
    1 youshang
    """
    cord_pre=np.zeros((4,2))
    cord_pre[0,1]=np.arctan((lines[1,0]*np.cos(lines[2,1])-lines[2,0]*np.cos(lines[1,1]))/
                            (lines[2,0]*np.sin(lines[1,1])-lines[1,0]*np.sin(lines[2,1])))
    cord_pre[0,0]=lines[1,0]/np.cos(cord_pre[0,1]-lines[1,1])

    cord_pre[1,1]=np.arctan((lines[3,0]*np.cos(lines[1,1])-lines[1,0]*np.cos(lines[3,1]))/
                            (lines[1,0]*np.sin(lines[3,1])-lines[3,0]*np.sin(lines[1,1])))
    cord_pre[1,0]=lines[3,0]/np.cos(cord_pre[1,1]-lines[3,1])

    cord_pre[2,1]=np.arctan((lines[0,0]*np.cos(lines[2,1])-lines[2,0]*np.cos(lines[0,1]))/
                            (lines[2,0]*np.sin(lines[0,1])-lines[0,0]*np.sin(lines[2,1])))
    cord_pre[2,0]=lines[2,0]/np.cos(cord_pre[2,1]-lines[2,1])

    cord_pre[3,1]=np.arctan((lines[0,0]*np.cos(lines[3,1])-lines[3,0]*np.cos(lines[0,1]))/
                            (lines[3,0]*np.sin(lines[0,1])-lines[0,0]*np.sin(lines[3,1])))
    cord_pre[3,0]=lines[3,0]/np.cos(cord_pre[3,1]-lines[3,1])


    cords=np.zeros((4,2))
    cords[:,0]=np.cos(cord_pre[:,1])*cord_pre[:,0]
    cords[:,1]=np.sin(cord_pre[:,1])*cord_pre[:,0]
    cords=np.int0(cords)
    return cords

def get_now_cord(srcedge,templateedge):
    res=cv2.matchTemplate(srcedge.copy(),templateedge,cv2.TM_CCOEFF_NORMED)
    _,_,_,max_loc=cv2.minMaxLoc(res)
    top_left=max_loc
    return(top_left[0]+15,top_left[1]+58)

def find_next_circle_cord(src_after_cut):
    img = src_after_cut
    img_blur=cv2.GaussianBlur(img,(5,5),0)
    #cv2.imwrite('img_cut.jpg',img_blur)
    gray=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    #edge=cv2.Canny(gray,50,250,apertureSize=5)
    edge=gray
    cv2.imwrite('cacacaca.jpg',edge)
    img=edge

    w=np.int0(img.shape[1]/1.6724)
    h=img.shape[0]
    pst1=np.float32([[img.shape[1],img.shape[0]],[img.shape[1],0],[0,img.shape[0]]])
    pst2=np.float32([[w,h],[w,0],[0,h]])               
    M=cv2.getAffineTransform(pst1,pst2)
    dst=cv2.warpAffine(img,M,(w,h))
    cv2.imwrite('dst2222.jpg',dst)
    
    circles=cv2.HoughCircles(dst,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=40)
    print(circles)
    if circles is None:
        return None
    circles=np.uint16(np.around(circles))
    circles=circles[0]
    print(circles)
    for i in circles[:]:
        cv2.circle(dst,(i[0],i[1]),2,255,3)
    cv2.imwrite('dstcir.jpg',dst)
        

    cords=circles[0][:2]
    cords[0]=np.int0(1.6724*cords[0])
    return cords

def get2cords(canny_src,place,now_cord):
    img=canny_src.copy()
    print(now_cord)
    if now_cord[0]>160:
        img[:,now_cord[0]:]=0
    else:
        img[:,:now_cord[0]]=0
    cv2.imwrite('rr.jpg',img)
    template=cv2.imread(place+'_cut.jpg',0)
    #template=cv2.GaussianBlur(template,(5,5),0)
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) 
    cv2.imwrite('res'+place+'.jpg',res*200)
    _,max_val,_,max_loc=cv2.minMaxLoc(res)
    cords=np.zeros((2,2))
    cords[0][0]=max_loc[0]
    cords[0][1]=max_loc[1]
    res[max_loc[1],max_loc[0]]=0
    while 1:
        _,max_val2,_,max_loc2=cv2.minMaxLoc(res)
        if place=='down':
            if max_loc2[0]<max_loc[0]+6 and max_loc2[0]>max_loc[0]-6:
                print(4)
                if max_loc2[1]<max_loc[1]-3 and max_loc2[1]>max_loc[1]-20:
                    res[max_loc2[1],max_loc2[0]]=0
                    continue
                elif max_loc[1]<max_loc2[1]-3 and max_loc[1]>max_loc2[1]-20:
                    if max_val2>0.7*max_val:
                        cords[0][0]=max_loc[0]
                        cords[0][1]=max_loc[1]
                        res[max_loc2[1],max_loc2[0]]=0
                        continue
                    else:
                        res[max_loc2[1],max_loc2[0]]=0
                        continue 
                else:
                    res[max_loc2[1],max_loc2[0]]=0
                    continue
            else: 
                cords[1][0]=max_loc2[0]
                cords[1][1]=max_loc2[1]
                break
                print(3)
        elif np.linalg.norm(list(max_loc2) - cords[0])<5:
            res[max_loc2[1],max_loc2[0]]=0
            print(max_loc2)
            continue
        cords[1][0]=max_loc2[0]
        cords[1][1]=max_loc2[1]
        print(1)
        break
    
    if place=='right':
        cords=cords+np.int0([21,11])
    if place=='left':
        cords=cords+np.int0([11,11])
    if place=='up':
        cords=cords+np.int0([21,10])
    if place=='down':
        cords=cords+np.int0([15,15])
    return np.int0(cords)



img = cv2.imread('0test4.jpg',1)#11  7
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('image.jpg',img)
cv2.imwrite('gray.jpg',gray)

_,thresh=cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
cv2.imwrite('thresh.jpg',thresh)
contours,hierarchy,chosen_index,cnt=finmaxcontour(thresh)
cv2.fillPoly(thresh,[cnt],(255,255,255))
cv2.imwrite('threshpol.jpg',thresh)

edge=cv2.Canny(thresh,50,150,apertureSize=5)
cv2.imwrite('edge.jpg',edge)
cords=find_4cords(edge)
print(cords) 

cords=np.float32(cords)
cords_new=np.float32([[0,0],[337,0],[0,600],[337,600]])

M=cv2.getPerspectiveTransform(cords,cords_new)

game_zone=cv2.warpPerspective(img,M,(337,600))
cv2.imwrite('game_zone.jpg',game_zone)
img=game_zone







templateedge=cv2.imread('templateedge.jpg',0)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,150,250,apertureSize=5)
now_cord=list(get_now_cord(edge,templateedge))

img_cut=img[100:now_cord[1],10:-10]
now_cord[1]=now_cord[1]-100
now_cord[0]=now_cord[0]-10
img=img_cut
#cv2.imwrite('use_for_square8.jpg',img)


circle_cord=find_next_circle_cord(img)
if circle_cord is None:
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,125,125,apertureSize=5) # 50 180
    up_cords=get2cords(edge,'up',now_cord)
    left_cords=get2cords(edge,'left',now_cord)
    right_cords=get2cords(edge,'right',now_cord)
    down_cords=get2cords(edge,'down',now_cord)
    
    
    cv2.circle(edge,tuple(up_cords[0]),3,255,2)
    cv2.circle(edge,tuple(up_cords[1]),1,255,2)
    cv2.imwrite('edge1.jpg',edge)
 
    cv2.circle(edge,tuple(left_cords[0]),3,255,2)
    cv2.circle(edge,tuple(left_cords[1]),1,255,2)
    cv2.imwrite('edge1.jpg',edge)
    
    cv2.circle(edge,tuple(down_cords[0]),3,255,2)
    cv2.circle(edge,tuple(down_cords[1]),1,255,2)
    cv2.imwrite('edge1.jpg',edge)    
    
    cv2.circle(edge,tuple(right_cords[0]),3,255,2)
    cv2.circle(edge,tuple(right_cords[1]),1,255,2)
    cv2.imwrite('edge1.jpg',edge)
    print(1515)
    print(edge.shape)
    print(up_cords[0])
    print(down_cords[0])
    print(left_cords[0])
    print(right_cords[0])
  
else:
    cv2.circle(img,(circle_cord[0],circle_cord[1]),2,(0,255,0),3)
    cv2.circle(img,(now_cord[0],now_cord[1]),2,(0,255,0),3)
    cv2.imwrite('testround2.jpg',img)    









"""
circle

back filter-----thresh
corner??

"""








