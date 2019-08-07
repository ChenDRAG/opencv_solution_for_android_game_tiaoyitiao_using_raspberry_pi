from picamera import PiCamera
import picamera.array
import time
import cv2
import numpy as np
"""
img = cv2.imread('game_zone.jpg',1)
template=cv2.imread('templateedge.jpg',0)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,150,250,apertureSize=5)
cv2.imwrite('1111.jpg',edge)

res=cv2.matchTemplate(edge,template,cv2.TM_CCOEFF_NORMED)
_,_,_,max_loc=cv2.minMaxLoc(res)
top_left=max_loc
w,h=template.shape[::-1]
bottom_right=(top_left[0]+w,top_left[1]+h)
cv2.circle(edge,(top_left[0]+15,top_left[1]+58),1,255,-1)
cv2.rectangle(edge,top_left,bottom_right,255,2)
cv2.imwrite('2222.jpg',edge)

"""

"""
def find_next_circle_cord(src_after_cut)
    img = src_after_cut
    img_blur=cv2.GaussianBlur(img,(5,5),0)
    #cv2.imwrite('img_cut.jpg',img_blur)
    gray=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,50,250,apertureSize=5)
    cv2.imwrite('cacacaca.jpg',edge)
    img=edge

    w=np.int0(img.shape[1]/1.6724)
    h=img.shape[0]
    pst1=np.float32([[img.shape[1],img.shape[0]],[img.shape[1],0],[0,img.shape[0]]])
    pst2=np.float32([[w,h],[w,0],[0,h]])               
    M=cv2.getAffineTransform(pst1,pst2)
    dst=cv2.warpAffine(img,M,(w,h))
    #cv2.imwrite('dst2222.jpg',dst)

    circles=cv2.HoughCircles(dst,cv2.HOUGH_GRADIENT,1,20,param1=10,param2=40)
    circles=np.uint16(np.around(circles))
    circles=circles[0]
    print(circles)
    for i in circles[:]:
        cv2.circle(dst,(i[0],i[1]),2,255,3)
    cv2.imwrite('dstcir.jpg',dst)
        

    cords=circles[0][:2]
    cords[0]=np.int0(1.6724*cords[0])
    return cords
    
cv2.circle(img,(cords[0],cords[1]),2,255,3)
img = cv2.imwrite('testround.jpg',img)
"""

"""
def get2cords(canny_src,place,now_cord):
    img=canny_src.copy()
    if now_cord[0]>160:
        img[now_cord[0]:]=0
    else:
        img[:now_cord[0]]=0
    template=cv2.imread(place+'_cut.jpg',0)
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) 
    cv2.imwrite('res'+place+'.jpg',res*200)
    _,max_val,_,max_loc=cv2.minMaxLoc(res)
    cords=np.zeros((2,2))
    cords[0][0]=max_loc[0]
    cords[0][1]=max_loc[1]
    res[max_loc]=0
    while 1:
        _,max_val2,_,max_loc2=cv2.minMaxLoc(res)
        if place=='down':
            if max_loc2[0]<max_loc[0]+6 and max_loc2[0]>max_loc[0]-6:
                if max_loc2[1]<max_loc[1]-3 and max_loc2[1]>max_loc[1]-20:
                    res[max_loc2]=0
                    continue
                else if max_loc[1]<max_loc2[1]-3 and max_loc[1]>max_loc2[1]-20:
                    if max_val2>0.7*max_val:
                        cords[0][0]=max_loc[0]
                        cords[0][1]=max_loc[1]
                        res[max_loc2]=0
                        continue
                    else:
                        res[max_loc2]=0
                        continue 
                else:
                    res[max_loc2]=0
                    continue
            else: 
                cords[1][0]=max_loc2[0]
                cords[1][1]=max_loc2[1]
                break
        else if numpy.linalg.norm(max_loc2 - max_loc)<10:
            res[max_loc2]=0
            continue
        cords[1][0]=max_loc2[0]
        cords[1][1]=max_loc2[1]
        break  
    if place=='right'
        cords=cords+np.int0([21,11])
    if place=='left'
        cords=cords+np.int0([11,11])
    if place=='up'
        cords=cords+np.int0([21,10])
    if place=='down'
        cords=cords+np.int0([15,15])
    return cords



img = cv2.imread('use_for_square1.jpg',1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,60,200,apertureSize=5)
    #right 21 11
    #down 15 15
    #up 21 10
    #left11 11
    
print(max_val)
print(max_loc)
res[max_loc[0],max_loc[1]]=0
_,max_val,_,max_loc=cv2.minMaxLoc(res)
print(max_val)
print(max_loc)
res[max_loc[0],max_loc[1]]=0


cord=max_loc+np.int0([11,11])
cv2.circle(edge,tuple(cord),1,255,2)
cv2.imwrite('gray.jpg',edge)    


"""
"""
#down gray 15 15
edge=cv2.imread('down_cut.jpg',0)
cv2.circle(edge,(15,15),3,255,5)
cv2.imwrite('gray.jpg',edge)
"""

"""
#cv2.imwrite('gray.jpg',edge)
k=edge[154:174,285:320]
cv2.circle(k,(18,13),0,255,2)
cv2.circle(k,(19,12),0,255,2)
#20 11


"""

"""
#top 20 10 
k=cv2.imread('gray_cut.jpg',0)

#k=cv2.GaussianBlur(k,(9,9),0)
#cv2.circle(k,(20,10),1,255,2)
#k_=cv2.add(k,k)
#k=cv2.add(k,k_)
np.set_printoptions(threshold=np.inf)
#print(k)
#cv2.imwrite('gray_cut_p.jpg',k)
res=cv2.matchTemplate(edge.copy(),k,cv2.TM_CCOEFF_NORMED)
cv2.imwrite('res.jpg',res*200)
print(np.max(res))
_,_,_,max_loc=cv2.minMaxLoc(res)
cord=max_loc+np.int0([20,10])
cv2.circle(edge,tuple(cord),1,255,3)
cv2.imwrite('gray.jpg',edge)
"""
def udfit(up_cord,down_cord):
    if down_cord[1]-up_cord[1]>15 and down_cord[1]-up_cord[1]< 113 and down_cord[0]-up_cord[0]>=-4 and down_cord[0]-up_cord[0]<=4:
        return True
    else:
        #print(up_cord)
        #print(down_cord)
        return False
    
def lrfit(left_cord,right_cord):
    if right_cord[0]-left_cord[0]>10 and right_cord[0]-left_cord[0]< 180 and right_cord[1]-left_cord[1]>=-4 and right_cord[1]-left_cord[1]<=4:
        #print(10)
        return True
    else:
        return False
def ur_ldfit(ul_cord,rd_cord):
    k=(ul_cord[1]-rd_cord[1])/(ul_cord[0]-rd_cord[0])
    if k<1.4 or k>1.8:
        return False
    delta=rd_cord[0]-ul_cord[0]
    if delta<5 or delta>90:
        return False
    return True
def ul_rdfit(ur_cord,ld_cord):
    k=(ur_cord[1]-ld_cord[1])/(ur_cord[0]-ld_cord[0])*(-1)
    if k<1.4 or k>1.8:
        return False
    delta=(ld_cord[0]-ur_cord[0])*(-1)
    if delta<5 or delta>90:
        return False
    return True

def fit3(up_cord,down_cord,left_cord,right_cord):
    if up_cord is None:
        if ur_ldfit(left_cord,down_cord)==False:
            return False
        if ul_rdfit(right_cord,down_cord)==False:
            return False
        if lrfit(left_cord,right_cord)==False:
            return False
        delta=np.abs((left_cord[0]+right_cord[0])/2-down_cord[0])
        if delta>5:
            return False
        return True
    if down_cord is None:
        if ur_ldfit(up_cord,right_cord)==False:
            return False
        if ul_rdfit(up_cord,left_cord)==False:
            return False
        if lrfit(left_cord,right_cord)==False:
            return False
        delta=np.abs((left_cord[0]+right_cord[0])/2-up_cord[0])
        if delta>5:
            return False
        return True
    if left_cord is None:
        if ur_ldfit(up_cord,right_cord)==False:
            return False
        if ul_rdfit(right_cord,down_cord)==False:
            return False
        if udfit(up_cord,down_cord)==False:
            return False
        delta=np.abs((up_cord[1]+down_cord[1])/2-right_cord[1])
        if delta>5:
            return False
        return True
    if right_cord is None:
        if ur_ldfit(left_cord,down_cord)==False:
            return False
        if ul_rdfit(up_cord,left_cord)==False:
            return False
        if udfit(up_cord,down_cord)==False:
            return False
        delta=np.abs((up_cord[1]+down_cord[1])/2-left_cord[1])
        if delta>5:
            return False
        return True


def fit4(up_cord,down_cord,left_cord,right_cord):
    if lrfit(left_cord,right_cord)==False or udfit(up_cord,down_cord)==False:
        return False
    rate=np.linalg.norm(left_cord-right_cord)/np.linalg.norm(up_cord-down_cord)
    if rate<1.5 or rate>1.81:
        return False 
    cordud=(up_cord+down_cord)/2
    cordlr=(left_cord+right_cord)/2
    if np.linalg.norm(cordud-cordlr)>10:
        return False
    else:
        #print(100)
        return True
        

def find_square_cord(up_cords,down_cords,left_cords,right_cords):
    cord=[0,0]
    #fit4
    for up_cord in up_cords:
        for right_cord in right_cords:
            for down_cord in down_cords:
                for left_cord in left_cords:
                    if fit4(up_cord,down_cord,left_cord,right_cord)==True:
                        cord=(up_cord+down_cord+left_cord+right_cord)/4.
                        return tuple(np.int0(cord))
    #fit3 no left
    for up_cord in up_cords:
        for right_cord in right_cords:
            for down_cord in down_cords:
                if fit3(up_cord,down_cord,None,right_cord)==True:
                    cord[0]=(up_cord+down_cord)/2
                    cord[1]=(up_cord+down_cord+right_cord)/3
                    return tuple(np.int0(cord))
    #fit3 no down
    for up_cord in up_cords:
        for right_cord in right_cords:
            for left_cord in left_cords:
                if fit3(up_cord,None,left_cord,right_cord)==True:
                    cord[0]=(up_cord+left_cord+right_cord)/3
                    cord[1]=(left_cord+right_cord)/2
                    return tuple(np.int0(cord))
    #fit 3 no right
    for up_cord in up_cords:
        for down_cord in down_cords:
            for left_cord in left_cords:
                if fit3(up_cord,down_cord,left_cord,None)==True:
                    cord[0]=(up_cord+down_cord)/2
                    cord[1]=(up_cord+down_cord+left_cord)/3
                    return tuple(np.int0(cord))                    
    #fit3 no up
    for right_cord in right_cords:
        for down_cord in down_cords:
            for left_cord in left_cords:  
                if fit3(None,down_cord,left_cord,right_cord)==True:
                    cord[0]=(down_cord+left_cord+right_cord)/3
                    cord[1]=(left_cord+right_cord)/2
                    return tuple(np.int0(cord))    
    return None
                    
                    
                    
                    
def fit2(up_cord,down_cord,left_cord,right_cord):
    if up_cord is None and down _cord is None:
        if lrfit(left_cord,right_cord)==False:
            return False
        return True
    if left_cord is None and right_cord is None:
        if udfit(up_cord,down_cord)==False:
            return False
        return True
        
        
      
    
    
        

