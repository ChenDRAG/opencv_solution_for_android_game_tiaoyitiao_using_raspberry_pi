from picamera import PiCamera
import picamera.array
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import serial    #import serial module
import time

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
    #cv2.imwrite('thresline.jpg',copy)
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
    #img_blur=cv2.GaussianBlur(img,(5,5),0)
    #cv2.imwrite('img_cut.jpg',img_blur)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,200,250,apertureSize=5)
    edge=gray
    #cv2.imwrite('cacacaca.jpg',edge)
    img=edge

    w=np.int0(img.shape[1]/1.6724)
    h=img.shape[0]
    pst1=np.float32([[img.shape[1],img.shape[0]],[img.shape[1],0],[0,img.shape[0]]])
    pst2=np.float32([[w,h],[w,0],[0,h]])               
    M=cv2.getAffineTransform(pst1,pst2)
    dst=cv2.warpAffine(img,M,(w,h))
    #cv2.imwrite('dst2222.jpg',dst)
    
    circles=cv2.HoughCircles(dst,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=30)
    #print(circles)
    if circles is None:
        return None
    circles=np.uint16(np.around(circles))
    circles=circles[0]
    #print(circles)
    for i in circles[:]:
        cv2.circle(dst,(i[0],i[1]),2,255,3)
    #cv2.imwrite('dstcir.jpg',dst)
        

    cords=circles[0][:2]
    cords[0]=np.int0(1.6724*cords[0])
    return cords

def get2cords(canny_src,place,now_cord):
    img=canny_src.copy()
    #print(now_cord)
    if now_cord[0]>160:
        img[:,now_cord[0]:]=0
    else:
        img[:,:now_cord[0]]=0
    #cv2.imwrite('rr.jpg',img)
    template=cv2.imread(place+'_cut.jpg',0)
    #template=cv2.GaussianBlur(template,(5,5),0)
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) 
    #cv2.imwrite('res'+place+'.jpg',res*200)
    _,max_val,_,max_loc=cv2.minMaxLoc(res)
    cords=np.zeros((2,2))
    cords[0][0]=max_loc[0]
    cords[0][1]=max_loc[1]
    res[max_loc[1],max_loc[0]]=0
    while 1:
        _,max_val2,_,max_loc2=cv2.minMaxLoc(res)
        if place=='down':
            if max_loc2[0]<max_loc[0]+6 and max_loc2[0]>max_loc[0]-6:
                #print(4)
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
                #print(3)
        elif np.linalg.norm(list(max_loc2) - cords[0])<5:
            res[max_loc2[1],max_loc2[0]]=0
            #print(max_loc2)
            continue
        cords[1][0]=max_loc2[0]
        cords[1][1]=max_loc2[1]
        #print(1)
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

def udfit(up_cord,down_cord):
    if down_cord[1]-up_cord[1]>15 and down_cord[1]-up_cord[1]< 113 and down_cord[0]-up_cord[0]>=-10 and down_cord[0]-up_cord[0]<=10:
        return True
    else:
        #print(up_cord)
        #print(down_cord)
        return False
    
def lrfit(left_cord,right_cord):
    if right_cord[0]-left_cord[0]>20 and right_cord[0]-left_cord[0]< 180 and right_cord[1]-left_cord[1]>=-10 and right_cord[1]-left_cord[1]<=10:
        #print(10)
        return True
    else:
        return False
    
    
def ur_ldfit(ul_cord,rd_cord):
    k=(ul_cord[0]-rd_cord[0])/(ul_cord[1]-rd_cord[1]+0.01)
    if k<1.4 or k>1.83:
        return False
    delta=rd_cord[0]-ul_cord[0]
    if delta<5 or delta>90:
        return False
    return True
def ul_rdfit(ur_cord,ld_cord):
    k=(ur_cord[0]-ld_cord[0])/(ur_cord[1]-ld_cord[1]+0.01)*(-1)
    if k<1.4 or k>1.83:
        return False
    delta=(ld_cord[0]-ur_cord[0])*(-1)
    if delta<5 or delta>90:
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
            #print(12)
            return False
        if ul_rdfit(up_cord,left_cord)==False:
            #print(13)
            return False
        if udfit(up_cord,down_cord)==False:
            #print(14)
            return False
        delta=np.abs((up_cord[1]+down_cord[1])/2-left_cord[1])
        if delta>5:
            #print(1)
            return False
        return True
                    
def fit2(up_cord,down_cord,left_cord,right_cord):
    if up_cord is None and down_cord is None:
        if lrfit(left_cord,right_cord)==False:
            return False
        return True
    if left_cord is None and right_cord is None:
        if udfit(up_cord,down_cord)==False:
            return False
        return True    

def find_square_cord(up_cords,down_cords,left_cords,right_cords):
    cord=np.zeros(2)
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
                    cord[0]=(up_cord[0]+down_cord[0])/2
                    cord[1]=(up_cord[1]+down_cord[1]+right_cord[1])/3
                    return tuple(np.int0(cord))
    #fit3 no down
    for up_cord in up_cords:
        for right_cord in right_cords:
            for left_cord in left_cords:
                if fit3(up_cord,None,left_cord,right_cord)==True:
                    cord[0]=(up_cord[0]+left_cord[0]+right_cord[0])/3
                    cord[1]=(left_cord[1]+right_cord[1])/2
                    return tuple(np.int0(cord))
    #fit 3 no right
    for up_cord in up_cords:
        for down_cord in down_cords:
            for left_cord in left_cords:
                if fit3(up_cord,down_cord,left_cord,None)==True:
                    cord[0]=(up_cord[0]+down_cord[0])/2
                    cord[1]=(up_cord[1]+down_cord[1]+left_cord[1])/3
                    return tuple(np.int0(cord))                    
    #fit3 no up
    for right_cord in right_cords:
        for down_cord in down_cords:
            for left_cord in left_cords:  
                if fit3(None,down_cord,left_cord,right_cord)==True:
                    cord[0]=(down_cord[0]+left_cord[0]+right_cord[0])/3
                    cord[1]=(left_cord[1]+right_cord[1])/2
                    return tuple(np.int0(cord))
                
    #fit 2
    for up_cord in up_cords:
        for down_cord in down_cords:
            if fit2(up_cord,down_cord,None,None)==True:
                cord[0]=(down_cord[0]+up_cord[0])/2
                cord[1]=(up_cord[1]+down_cord[1])/2
                return tuple(np.int0(cord))
    for right_cord in right_cords:
         for left_cord in left_cords:
            if fit2(None,None,left_cord,right_cord)==True:
                cord[0]=(left_cord[0]+right_cord[0])/2
                cord[1]=(left_cord[1]+right_cord[1])/2
                return tuple(np.int0(cord))
                
            
    return None


def pre_sort(src):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('image.jpg',img)
    #cv2.imwrite('gray.jpg',gray)

    _,thresh=cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
    #cv2.imwrite('thresh.jpg',thresh)
    contours,hierarchy,chosen_index,cnt=finmaxcontour(thresh)
    cv2.fillPoly(thresh,[cnt],(255,255,255))
    #cv2.imwrite('threshpol.jpg',thresh)

    edge=cv2.Canny(thresh,50,150,apertureSize=5)
    return edge

def find_game_zone(src):
    img=src
    edge=pre_sort(img)
    #cv2.imwrite('edge.jpg',edge)
    cords=find_4cords(edge)
    #print(cords) 

    cords=np.float32(cords)
    cords_new=np.float32([[0,0],[337,0],[0,600],[337,600]])

    M=cv2.getPerspectiveTransform(cords,cords_new)

    game_zone=cv2.warpPerspective(img,M,(337,600))
    return game_zone

flag=1
cam=PiCamera()
ser = serial.Serial('/dev/ttyUSB0', 9600,timeout=1)
while True:
    cam.start_preview()
    if flag==1:
        time.sleep(5)
        flag=0
    time.sleep(2)
    cam.capture('img_original.jpg')
    cam.stop_preview()
    img = cv2.imread('img_original.jpg',1)#

    game_zone=find_game_zone(img)
    cv2.imwrite('game_zone.jpg',game_zone)
    img=game_zone



    templateedge=cv2.imread('templateedge.jpg',0)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,150,250,apertureSize=5)
    now_cord=list(get_now_cord(edge,templateedge))

    img_cut=img[100:now_cord[1],10:-10]
    now_cord[1]=now_cord[1]-100
    now_cord[0]=now_cord[0]-10
    now_cord=tuple(now_cord)
    img=img_cut
    cv2.imwrite('img_cut.jpg',img)


    next_cord=np.zeros(2)
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,115,120,apertureSize=5) # 50 180
    up_cords=get2cords(edge,'up',now_cord)
    left_cords=get2cords(edge,'left',now_cord)
    right_cords=get2cords(edge,'right',now_cord)
    down_cords=get2cords(edge,'down',now_cord)
    
    tem=edge.copy()
    cv2.circle(edge,tuple(up_cords[0]),3,255,2)
    cv2.circle(edge,tuple(up_cords[1]),1,255,2)
 
    cv2.circle(edge,tuple(left_cords[0]),3,255,2)
    cv2.circle(edge,tuple(left_cords[1]),1,255,2)
    
    cv2.circle(edge,tuple(down_cords[0]),3,255,2)
    cv2.circle(edge,tuple(down_cords[1]),1,255,2)  
    
    cv2.circle(edge,tuple(right_cords[0]),3,255,2)
    cv2.circle(edge,tuple(right_cords[1]),1,255,2)
    cv2.imwrite('edge.jpg',edge)
    edge=tem
    """
    print(1515)
    print(edge.shape)

    print(up_cords[0])
    print(down_cords[0])
    print(left_cords[0])
    print(right_cords[0])
    """
    """
    print(up_cords)
    print(down_cords)
    print(left_cords)
    print(right_cords)
     """
    square_cord=find_square_cord(up_cords,down_cords,left_cords,right_cords)
    #print(str(i)+'ts')
    if square_cord is None:        
        circle_cord=find_next_circle_cord(img)
        if circle_cord is None:
            tem=(right_cords+left_cords+up_cords+down_cords)/4
            next_cord=(int((tem[1][0]+tem[0][0])/2),int((tem[1][1]+tem[0][1])/2))
            print('Failure')
        else:
            print('CIRCLE')
            next_cord=(circle_cord[0],circle_cord[1])
            #cv2.imwrite('testround2.jpg',img)
    else:
        cv2.circle(edge,square_cord,5,255,2)
        next_cord=square_cord
        print('SQURE')
    #cv2.imwrite('edge'+str(i)+'.jpg',edge)        
    
    cv2.circle(edge,next_cord,2,255,3)
    cv2.circle(edge,now_cord,2,255,3)
    distance=np.linalg.norm(np.int0(now_cord)-np.int0(next_cord))
    cv2.imwrite('outcome.jpg',edge)
    print('*********')
    print('now')
    print(now_cord)
    print('next')
    print(next_cord)
    print('Dis')
    print(distance)
    print('*********')

    time_ = int(distance*4.44+91.7)
    time_=str(time_)
    print('time:')
    print(time_)
    tem=input('ano1')
    ser.write(time_.encode())
    time.sleep(0.5)
    while True:
        response = ser.readline()#read a string from port
        if response !=b'' and response !=b'\n':
            break
    print(str(response))
    time.sleep(0.5)
    while True:
        response = ser.readline()
        print(response)
        if response==b'Done\r\n':
            break
    tem=input('ano2')




    
    
    
    
    
    
    
        
        










