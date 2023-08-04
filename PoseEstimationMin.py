from cgitb import reset
from re import T
from sre_constants import SUCCESS
from tkinter import Image
from unittest import result
import cv2
import mediapipe as mp
import time

from numpy import append, dot
import numpy as np

#L2 정규화
def L2(inp):
    result = 0
    for i in inp:
        result += i ** 2
    return np.sqrt(result)

#코사인 유사도
def cos_sim(inp1, inp2):
    return dot(inp1, inp2)/(L2(inp1) * L2(inp2))

#점수화
def score(inp1,inp2):

    cs = 2*(1-cos_sim(inp1,inp2))
    if cs <0:
        cs= -cs
    return 100-(50*(np.sqrt(cs)))
   
    

#시범자 부위별 데이터를 담을 리스트 생성
video_shoulder = []  # 양쪽 어깨 11 to 12
video_left_top_arm =[] # 왼쪽 윗팔 12 to 14
video_right_top_arm = [] #오른쪽 윗팔 11 to 13 
video_left_down_arm = [] #왼쪽 아래팔 14 to 16
video_right_down_arm = [] #오른쪽 아래팔 13 to 15
video_left_body = [] #왼쪽 옆구리 12 to 24 
video_right_body = [] #오른쪽 옆구리 11 to 23
video_hip = [] #양쪽 골반 23 to 24
video_left_top_leg = [] #왼쪽 윗다리 24 to 26 
video_right_top_leg = [] #오른쪽 윗다리 23 to 25
video_left_down_leg = [] #왼쪽 아래다리 26 to 28
video_right_down_leg = [] #오른쪽 아래다리 25 to 27

#참여자 부위별 데이터를 담을 리스트 생성
cam_shoulder = []  # 양쪽 어깨 11 to 12
cam_left_top_arm =[] # 왼쪽 윗팔 12 to 14
cam_right_top_arm = [] #오른쪽 윗팔 11 to 13 
cam_left_down_arm = [] #왼쪽 아래팔 14 to 16
cam_right_down_arm = [] #오른쪽 아래팔 13 to 15
cam_left_body = [] #왼쪽 옆구리 12 to 24 
cam_right_body = [] #오른쪽 옆구리 11 to 23
cam_hip = [] #양쪽 골반 23 to 24
cam_left_top_leg = [] #왼쪽 윗다리 24 to 26 
cam_right_top_leg = [] #오른쪽 윗다리 23 to 25
cam_left_down_leg = [] #왼쪽 아래다리 26 to 28
cam_right_down_leg = [] #오른쪽 아래다리 25 to 27


mpDraw = mp.solutions.drawing_utils
mpDraw2 = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mpPose2 = mp.solutions.pose
pose=mpPose.Pose()
pose2=mpPose2.Pose()

cap = cv2.VideoCapture('PoseVideos/solo6_Trim.mp4')
capture = cv2.VideoCapture('PoseVideos/solo5_Trim.mp4')

while True:
    SUCCESS, me = capture.read()
    if  me is None:
         break
    #me= cv2.resize(me, (480,640))
    imgRGB2=cv2.cvtColor(me,cv2.COLOR_BGR2RGB)
    results2=pose2.process(imgRGB2)
    #print(results.pose_landmarks)
    if results2.pose_landmarks:
        mpDraw2.draw_landmarks(me,results2.pose_landmarks
                              ,mpPose2.POSE_CONNECTIONS)  
    #참여자의 각 부위별 좌표를 저장
    
    try:
        poseData= []
        for id, lm in enumerate(results2.pose_landmarks.landmark):
            h, w, c = me.shape
            data = np.array([lm.x, lm.y])
            poseData.append(data)
            
        cam_shoulder.append(poseData[11]-poseData[12])  # 양쪽 어깨 11 to 12
        cam_left_top_arm.append(poseData[12]-poseData[14]) # 왼쪽 윗팔 12 to 14
        cam_right_top_arm.append(poseData[11]-poseData[13]) #오른쪽 윗팔 11 to 13 
        cam_left_down_arm.append(poseData[14]-poseData[16]) #왼쪽 아래팔 14 to 16
        cam_right_down_arm.append(poseData[13]-poseData[15]) #오른쪽 아래팔 13 to 15
        cam_left_body.append(poseData[12]-poseData[24]) #왼쪽 옆구리 12 to 24 
        cam_right_body.append(poseData[11]-poseData[23]) #오른쪽 옆구리 11 to 23
        cam_hip.append(poseData[23]-poseData[24]) #양쪽 골반 23 to 24
        cam_left_top_leg.append(poseData[24]-poseData[26]) #왼쪽 윗다리 24 to 26 
        cam_right_top_leg.append(poseData[23]-poseData[25]) #오른쪽 윗다리 23 to 25
        cam_left_down_leg.append(poseData[26]-poseData[28]) #왼쪽 아래다리 26 to 28
        cam_right_down_leg.append(poseData[25]-poseData[27]) #오른쪽 아래다리 25 to 27
        poseData.clear()   
                
    #예외처리(참여자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        cam_shoulder.append(np.array([0,0]))  # 양쪽 어깨 11 to 12
        cam_left_top_arm.append(np.array([0,0])) # 왼쪽 윗팔 12 to 14
        cam_right_top_arm.append(np.array([0,0])) #오른쪽 윗팔 11 to 13 
        cam_left_down_arm.append(np.array([0,0])) #왼쪽 아래팔 14 to 16
        cam_right_down_arm.append(np.array([0,0])) #오른쪽 아래팔 13 to 15
        cam_left_body.append(np.array([0,0])) #왼쪽 옆구리 12 to 24 
        cam_right_body.append(np.array([0,0])) #오른쪽 옆구리 11 to 23
        cam_hip.append(np.array([0,0])) #양쪽 골반 23 to 24
        cam_left_top_leg.append(np.array([0,0])) #왼쪽 윗다리 24 to 26 
        cam_right_top_leg.append(np.array([0,0])) #오른쪽 윗다리 23 to 25
        cam_left_down_leg.append(np.array([0,0])) #왼쪽 아래다리 26 to 28
        cam_right_down_leg.append(np.array([0,0])) #오른쪽 아래다리 25 to 27
print(len(cam_hip))        
while True:
    SUCCESS, img = cap.read()
    if  img is None:
         break
    img = cv2.flip(img,1) #좌우반전
    #img= cv2.resize(img, (480,640))
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks
                              ,mpPose.POSE_CONNECTIONS)
    
    #시범자의 각 부위별 좌표를 저장
    try:
        poseData= []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            data = np.array([lm.x, lm.y])
            poseData.append(data)
       
        video_shoulder.append(poseData[11]-poseData[12])  # 양쪽 어깨 11 to 12
        video_left_top_arm.append(poseData[12]-poseData[14]) # 왼쪽 윗팔 12 to 14
        video_right_top_arm.append(poseData[11]-poseData[13]) #오른쪽 윗팔 11 to 13 
        video_left_down_arm.append(poseData[14]-poseData[16]) #왼쪽 아래팔 14 to 16
        video_right_down_arm.append(poseData[13]-poseData[15]) #오른쪽 아래팔 13 to 15
        video_left_body.append(poseData[12]-poseData[24]) #왼쪽 옆구리 12 to 24 
        video_right_body.append(poseData[11]-poseData[23]) #오른쪽 옆구리 11 to 23
        video_hip.append(poseData[23]-poseData[24]) #양쪽 골반 23 to 24
        video_left_top_leg.append(poseData[24]-poseData[26]) #왼쪽 윗다리 24 to 26 
        video_right_top_leg.append(poseData[23]-poseData[25]) #오른쪽 윗다리 23 to 25
        video_left_down_leg.append(poseData[26]-poseData[28]) #왼쪽 아래다리 26 to 28
        video_right_down_leg.append(poseData[25]-poseData[27]) #오른쪽 아래다리 25 to 27
        poseData.clear()
        
    #예외처리(시범자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        video_shoulder.append(np.array([0,0]))  # 양쪽 어깨 11 to 12
        video_left_top_arm.append(np.array([0,0])) # 왼쪽 윗팔 12 to 14
        video_right_top_arm.append(np.array([0,0])) #오른쪽 윗팔 11 to 13 
        video_left_down_arm.append(np.array([0,0])) #왼쪽 아래팔 14 to 16
        video_right_down_arm.append(np.array([0,0])) #오른쪽 아래팔 13 to 15
        video_left_body.append(np.array([0,0])) #왼쪽 옆구리 12 to 24 
        video_right_body.append(np.array([0,0])) #오른쪽 옆구리 11 to 23
        video_hip.append(np.array([0,0])) #양쪽 골반 23 to 24
        video_left_top_leg.append(np.array([0,0])) #왼쪽 윗다리 24 to 26 
        video_right_top_leg.append(np.array([0,0])) #오른쪽 윗다리 23 to 25
        video_left_down_leg.append(np.array([0,0])) #왼쪽 아래다리 26 to 28
        video_right_down_leg.append(np.array([0,0])) #오른쪽 아래다리 25 to 27
    
           


  
#유사도 비교 (임시)       
for i in range(len(cam_shoulder)):
    print(i,"번째 프레임")
    if cam_shoulder[i][0] ==0:
        print("검출되지 않음")
    else:
        score_list=[]
        for j in range(10):
            sum=0
            if i < 5 :
                if video_shoulder[j][0] ==0:
                    score_list.append(0)
                else:
                    #양쪽 어깨 
                    sim_score = score(video_shoulder[i],cam_shoulder[j])
                    sum +=sim_score
        
                    #왼쪽 윗팔
                    sim_score = score(video_left_top_arm[i],cam_left_top_arm[j])
                    sum +=sim_score
        
                    #오른쪽 윗팔
                    sim_score = score(video_right_top_arm[i],cam_right_top_arm[j])
                    sum +=sim_score
        
                    #왼쪽 아래팔
                    sim_score = score(video_left_down_arm[i],cam_left_down_arm[j])
                    sum +=sim_score
        
                    #오른쪽 아래팔
                    sim_score = score(video_right_down_arm[i],cam_right_down_arm[j])
                    sum +=sim_score
        
                    #왼쪽 옆구리
                    sim_score = score(video_left_body[i],cam_left_body[j])
                    sum +=sim_score
        
                    #오른쪽 옆구리
                    sim_score = score(video_right_body[i],cam_right_body[j])
                    sum +=sim_score
        
                    #양쪽 골반
                    sim_score = score(video_hip[i],cam_hip[j])
                    sum +=sim_score
        
                    #왼쪽 윗다리
                    sim_score = score(video_left_top_leg[i],cam_left_top_leg[j])
                    sum +=sim_score
        
                    #오른쪽 윗다리
                    sim_score = score(video_right_top_leg[i],cam_right_top_leg[j])
                    sum +=sim_score
        
                    #왼쪽 아래다리
                    sim_score = score(video_left_down_leg[i],cam_left_down_leg[j])
                    sum +=sim_score
        
                    #오른쪽 아래다리
                    sim_score = score(video_right_down_leg[i],cam_right_down_leg[j])
                    sum +=sim_score
                                       
            else:
                if video_shoulder[i-5+j][0] ==0:
                    score_list.append(0)
                else:
                    #양쪽 어깨 
                    sim_score = score(video_shoulder[i],cam_shoulder[i-5+j])
                    sum +=sim_score
        
                    #왼쪽 윗팔
                    sim_score = score(video_left_top_arm[i],cam_left_top_arm[i-5+j])
                    sum +=sim_score
        
                    #오른쪽 윗팔
                    sim_score = score(video_right_top_arm[i],cam_right_top_arm[i-5+j])
                    sum +=sim_score
        
                    #왼쪽 아래팔
                    sim_score = score(video_left_down_arm[i],cam_left_down_arm[i-5+j])
                    sum +=sim_score
        
                    #오른쪽 아래팔
                    sim_score = score(video_right_down_arm[i],cam_right_down_arm[i-5+j])
                    sum +=sim_score
        
                    #왼쪽 옆구리
                    sim_score = score(video_left_body[i],cam_left_body[i-5+j])
                    sum +=sim_score
        
                    #오른쪽 옆구리
                    sim_score = score(video_right_body[i],cam_right_body[i-5+j])
                    sum +=sim_score
        
                    #양쪽 골반
                    sim_score = score(video_hip[i],cam_hip[i-5+j])
                    sum +=sim_score
        
                    #왼쪽 윗다리
                    sim_score = score(video_left_top_leg[i],cam_left_top_leg[i-5+j])
                    sum +=sim_score
        
                    #오른쪽 윗다리
                    sim_score = score(video_right_top_leg[i],cam_right_top_leg[i-5+j])
                    sum +=sim_score
        
                    #왼쪽 아래다리
                    sim_score = score(video_left_down_leg[i],cam_left_down_leg[i-5+j])
                    sum +=sim_score
        
                    #오른쪽 아래다리
                    sim_score = score(video_right_down_leg[i],cam_right_down_leg[i-5+j])
                    sum +=sim_score
        score_list.append(sum/12)
        tmp = max(score_list)
        index = score_list.index(tmp)
        sum=0
        #양쪽 어깨 
        sim_score = score(video_shoulder[index],cam_shoulder[i])
        print("양쪽 어깨 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 윗팔
        sim_score = score(video_left_top_arm[index],cam_left_top_arm[i])
        print("왼쪽 윗팔 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 윗팔
        sim_score = score(video_right_top_arm[index],cam_right_top_arm[i])
        print("오른쪽 윗팔 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 아래팔
        sim_score = score(video_left_down_arm[index],cam_left_down_arm[i])
        print("왼쪽 아래팔 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 아래팔
        sim_score = score(video_right_down_arm[index],cam_right_down_arm[i])
        print("오른쪽 아래팟 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 옆구리
        sim_score = score(video_left_body[index],cam_left_body[i])
        print("왼쪽 옆구리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 옆구리
        sim_score = score(video_right_body[index],cam_right_body[i])
        print("오른쪽 옆구리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #양쪽 골반
        sim_score = score(video_hip[index],cam_hip[i])
        print("양쪽 골반 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 윗다리
        sim_score = score(video_left_top_leg[index],cam_left_top_leg[i])
        print("왼쪽 윗다리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 윗다리
        sim_score = score(video_right_top_leg[index],cam_right_top_leg[i])
        print("오른쪽 윗다리 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 아래다리
        sim_score = score(video_left_down_leg[index],cam_left_down_leg[i])
        print("왼쪽 아래 다리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 아래다리
        sim_score = score(video_right_down_leg[index],cam_right_down_leg[i])
        print("오른쪽 아래다리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        print("평균 점수 : " ,sum/12,"\n")    
   

print("종료")
cv2.destroyAllWindows()    


    