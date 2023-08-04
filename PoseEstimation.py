from decimal import MAX_EMAX
from unittest.mock import CallableMixin
import webbrowser
from cv2 import CAP_PROP_XI_ACQ_TIMING_MODE, imshow
import numpy as np
from xml.etree.ElementTree import tostring
import cv2 as cv
import cv2
from cv2 import norm
import mediapipe as mp
import time
from numpy import dot
mpDrawCam = mp.solutions.drawing_utils
mpDrawVideo = mp.solutions.drawing_utils
mpPoseCam = mp.solutions.pose
mpPoseVideo = mp.solutions.pose
posecam = mpPoseCam.Pose()
posevideo = mpPoseVideo.Pose()



def Cropping(VideopointsX,VideopointsY,IMG):
    
    maxX=max(VideopointsX[11],VideopointsX[12],VideopointsX[23],VideopointsX[24])
    minX=min(VideopointsX[11],VideopointsX[12],VideopointsX[23],VideopointsX[24])
    
    #11,12,23,24 번 좌표들로 이루어진 다각형의 무게중심을 구함.
    #centerX=(VideopointsX[23] + (( maxX- minX) / 2))   
    #centerY = min(VideopointsY[11],VideopointsY[12]) + ((max(VideopointsY[23],VideopointsY[24]) - min(VideopointsY[11],VideopointsY[12])) / 2)
 
    centerX=(VideopointsX[23]+VideopointsX[24])/2
    centerY=(VideopointsY[23]+VideopointsY[24])/2
    
    
    
    
        
    #제일 작은 x y 와 제일 큰 x y 를 모두 구함   
    maxX=max(VideopointsX)
    minX=min(VideopointsX)
    maxY=max(VideopointsY)
    minY=min(VideopointsY)
    #무게중심 좌표로 부터 가장 큰 차이를 구함.
    RectRadius=max(maxX-centerX,maxY-centerY,centerX-minX,centerY-minY)
    
    #가장 큰 차이로 만든 정사각형 모양의 새로운 Frame 출력 , 그 정사각형의 시작 좌표 출력.    
    return IMG[int(centerY-RectRadius):int(centerY+RectRadius),int(centerX-RectRadius):int(centerX+RectRadius)],centerX-RectRadius,centerY-RectRadius,centerX+RectRadius,centerY+RectRadius
    
    
def L2(inp):
    result = 0
    for i in inp:
        result += i ** 2
    return np.sqrt(result)

def cos_sim(inp1, inp2):
    return dot(inp1, inp2)/(L2(inp1) * L2(inp2))             
    

frameCnt=0
falseCnt=0
     
capVideo = cv2.VideoCapture('PoseVideos/solo5.mp4')
capCam = cv2.VideoCapture('PoseVideos/solo6.mp4')
pTime = 0

while True:
    Videosuccess, Videoimg = capVideo.read()
    Camsuccess, Camimg = capCam.read()
    Camimg = cv2.flip(Camimg, 1)

    VideoimgRGB = cv2.cvtColor(Videoimg, cv2.COLOR_BGR2RGB)
    CamimgRGB = cv2.cvtColor(Camimg, cv2.COLOR_BGR2RGB)
    
    Videoresults = posevideo.process(VideoimgRGB)
    Camresults = posecam.process(CamimgRGB)

    WebCampointsX=[]
    WebCampointsY=[]
    VideopointsX=[]
    VideopointsY=[]
    
    VideopointsX.clear()
    VideopointsY.clear() 
    WebCampointsY.clear()
    WebCampointsX.clear()   
    count=0     
    if Camresults.pose_landmarks and Videoresults.pose_landmarks:
        mpDrawCam.draw_landmarks(Camimg, Camresults.pose_landmarks, mpPoseCam.POSE_CONNECTIONS)
        mpDrawVideo.draw_landmarks(Videoimg, Videoresults.pose_landmarks, mpPoseVideo.POSE_CONNECTIONS)   
        for id, lm in enumerate(Camresults.pose_landmarks.landmark):
            
            h, w, c = Camimg.shape
            hV, wV, cV = Videoimg.shape
            lmv=Videoresults.pose_landmarks.landmark[id]
            
            cx, cy = int(lm.x * w)/5+100, int(lm.y * h)/5+100
            vx, vy = int(lmv.x * wV), int(lmv.y * hV)

            #cv2.putText(Camimg, "{}".format(id), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            #           lineType=cv2.LINE_AA)
             
            VideopointsX.append(vx)
            VideopointsY.append(vy)
            WebCampointsX.append(cx)
            WebCampointsY.append(cy)
            
        
             
        #크롭 해서 크롭된 사각형의 제일 작은 x,y좌표와 크롭된 Frame 리턴.
        
        
        croppedVideoimg,VideoimgCroppedX,VideoimgCroppedY,VideoimglastX,VideoimglastY=Cropping(VideopointsX,VideopointsY,Videoimg) 
       
        croppedWebcamimg,WebcamCroppedX,WebcamCroppedY,WebcamlastX,WebcamlastY=Cropping(WebCampointsX,WebCampointsY,Camimg)
        
        
        
        
        Camimg=cv2.rectangle(Camimg,(int(WebcamCroppedX),int(WebcamCroppedY)),(int(WebcamlastX),int(WebcamlastY)),(255,255,0),3)
        Videoimg=cv2.rectangle(Videoimg,(int(VideoimgCroppedX),int(VideoimgCroppedY)),(int(VideoimglastX),int(VideoimglastY)),(255,255,0),3)


        sum =0
        for id, lm in enumerate(Camresults.pose_landmarks.landmark):    
            if ( id>10 and not(id>16 and id<23) and not(id>28)):
                vx=VideopointsX[id]-VideoimgCroppedX
                vy=VideopointsY[id]-VideoimgCroppedY
                cx=WebCampointsX[id]-WebcamCroppedX
                cy=WebCampointsY[id]-WebcamCroppedY
            
            
                #관절 좌표마다 파랑색 점을 찍음.
                cv2.circle(croppedVideoimg, (int(vx), int(vy)), 5, (255, 0, 0), cv2.FILLED)
                #관절 좌표 주위에 관절 번호를 찍음.(=id(0~33))
                cv2.putText(croppedVideoimg, "{}".format(id), (int(vx), int(vy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)
            
                cv2.circle(croppedWebcamimg, (int(cx), int(cy)), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(croppedWebcamimg, "{}".format(id), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)
            
                #각 좌표마다 코사인 유사도 구해서 출력.
                #print(str(id)+"번좌표")
            
                sum +=((cos_sim((np.array([cx,cy])),np.array([vx,vy]))*100)-90)*10
                #print(cos_sim((np.array([cx,cy])),np.array([vx,vy]))*100)
            
        
            
            
        #print(str(WebCampointsX[24]-WebcamCroppedX))    
        print(sum/12)
        if(sum/12<90):
            falseCnt+=1
            cv2.waitKey(1000)
        sum=0
        frameCnt+=1
        print(frameCnt,falseCnt)
        
        #croppedVideoimg=Videoimg        
        
     
        cv2.imshow("Video",Videoimg)
        cv2.imshow("Cam",  Camimg )
        cv2.waitKey(1)
