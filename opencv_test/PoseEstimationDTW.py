from cgitb import reset
from re import T
from sre_constants import SUCCESS
from tkinter import Image
from unittest import result
import cv2
import mediapipe as mp
import time
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from dtaidistance import dtw_ndim
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
video_shoulder = np.empty((0,2), dtype=np.double)  # 양쪽 어깨 11 to 12
video_left_top_arm =np.empty((0,2), dtype=np.double) # 왼쪽 윗팔 12 to 14
video_right_top_arm = np.empty((0,2), dtype=np.double) #오른쪽 윗팔 11 to 13 
video_left_down_arm = np.empty((0,2), dtype=np.double) #왼쪽 아래팔 14 to 16
video_right_down_arm = np.empty((0,2), dtype=np.double) #오른쪽 아래팔 13 to 15
video_left_body = np.empty((0,2), dtype=np.double) #왼쪽 옆구리 12 to 24 
video_right_body = np.empty((0,2), dtype=np.double) #오른쪽 옆구리 11 to 23
video_hip = np.empty((0,2), dtype=np.double) #양쪽 골반 23 to 24
video_left_top_leg = np.empty((0,2), dtype=np.double) #왼쪽 윗다리 24 to 26 
video_right_top_leg = np.empty((0,2), dtype=np.double) #오른쪽 윗다리 23 to 25
video_left_down_leg = np.empty((0,2), dtype=np.double) #왼쪽 아래다리 26 to 28
video_right_down_leg = np.empty((0,2), dtype=np.double) #오른쪽 아래다리 25 to 27

#참여자 부위별 데이터를 담을 리스트 생성
cam_shoulder =np.empty((0,2), dtype=np.double)  # 양쪽 어깨 11 to 12
cam_left_top_arm =np.empty((0,2), dtype=np.double) # 왼쪽 윗팔 12 to 14
cam_right_top_arm = np.empty((0,2), dtype=np.double) #오른쪽 윗팔 11 to 13 
cam_left_down_arm = np.empty((0,2), dtype=np.double) #왼쪽 아래팔 14 to 16
cam_right_down_arm = np.empty((0,2), dtype=np.double) #오른쪽 아래팔 13 to 15
cam_left_body = np.empty((0,2), dtype=np.double) #왼쪽 옆구리 12 to 24 
cam_right_body = np.empty((0,2), dtype=np.double) #오른쪽 옆구리 11 to 23
cam_hip = np.empty((0,2), dtype=np.double) #양쪽 골반 23 to 24
cam_left_top_leg = np.empty((0,2), dtype=np.double) #왼쪽 윗다리 24 to 26 
cam_right_top_leg = np.empty((0,2), dtype=np.double) #오른쪽 윗다리 23 to 25
cam_left_down_leg =np.empty((0,2), dtype=np.double) #왼쪽 아래다리 26 to 28
cam_right_down_leg = np.empty((0,2), dtype=np.double) #오른쪽 아래다리 25 to 27


mpDraw = mp.solutions.drawing_utils
mpDraw2 = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mpPose2 = mp.solutions.pose
pose=mpPose.Pose()
pose2=mpPose2.Pose()

cap = cv2.VideoCapture('PoseVideos/solo6_Trim.mp4')
capture = cv2.VideoCapture('PoseVideos/solo6.mp4')


while True:
    SUCCESS, img = cap.read()
    if  img is None:
         break
    #img = cv2.flip(img,1) #좌우반전
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
        video_shoulder = np.append(video_shoulder, np.array([poseData[11]-poseData[12]]), axis=0)# 양쪽 어깨 11 to 12
        video_left_top_arm = np.append(video_left_top_arm, np.array([poseData[12]-poseData[14]]), axis=0)# 왼쪽 윗팔 12 to 14
        video_right_top_arm = np.append(video_right_top_arm, np.array([poseData[11]-poseData[13]]), axis=0)#오른쪽 윗팔 11 to 13 
        video_left_down_arm = np.append(video_left_down_arm, np.array([poseData[14]-poseData[16]]), axis=0)#왼쪽 아래팔 14 to 16
        video_right_down_arm = np.append(video_right_down_arm, np.array([poseData[13]-poseData[15]]), axis=0)#오른쪽 아래팔 13 to 15
        video_left_body = np.append(video_left_body, np.array([poseData[12]-poseData[24]]), axis=0)#왼쪽 옆구리 12 to 24 
        video_right_body = np.append(video_right_body, np.array([poseData[11]-poseData[23]]), axis=0)#오른쪽 옆구리 11 to 23
        video_hip = np.append(video_hip, np.array([poseData[23]-poseData[24]]), axis=0)#양쪽 골반 23 to 24
        video_left_top_leg = np.append(video_left_top_leg, np.array([poseData[24]-poseData[26]]), axis=0)#왼쪽 윗다리 24 to 26 
        video_right_top_leg = np.append(video_right_top_leg, np.array([poseData[23]-poseData[25]]), axis=0)#오른쪽 윗다리 23 to 25
        video_left_down_leg = np.append(video_left_down_leg, np.array([poseData[26]-poseData[28]]), axis=0)#왼쪽 아래다리 26 to 28
        video_right_down_leg = np.append(video_right_down_leg, np.array([poseData[25]-poseData[27]]), axis=0)#오른쪽 아래다리 25 to 27

        poseData.clear()
        
    #예외처리(시범자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        print(id)
    """_summary_
    
        video_shoulder = np.append(video_shoulder, np.array([[0,0]]), axis=0)# 양쪽 어깨 11 to 12
        video_left_top_arm = np.append(video_left_top_arm, np.array([[0,0]]), axis=0)# 왼쪽 윗팔 12 to 14
        video_right_top_arm = np.append(video_right_top_arm, np.array([[0,0]]), axis=0)#오른쪽 윗팔 11 to 13 
        video_left_down_arm = np.append(video_left_down_arm, np.array([[0,0]]), axis=0)#왼쪽 아래팔 14 to 16
        video_right_down_arm = np.append(video_right_down_arm, np.array([[0,0]]), axis=0)#오른쪽 아래팔 13 to 15
        video_left_body = np.append(video_left_body, np.array([[0,0]]), axis=0)#왼쪽 옆구리 12 to 24 
        video_right_body = np.append(video_right_body, np.array([[0,0]]), axis=0)#오른쪽 옆구리 11 to 23
        video_hip = np.append(video_hip,np.array([[0,0]]), axis=0)#양쪽 골반 23 to 24
        video_left_top_leg = np.append(video_left_top_leg, np.array([[0,0]]), axis=0)#왼쪽 윗다리 24 to 26 
        video_right_top_leg = np.append(video_right_top_leg, np.array([[0,0]]), axis=0)#오른쪽 윗다리 23 to 25
        video_left_down_leg = np.append(video_left_down_leg, np.array([[0,0]]), axis=0)#왼쪽 아래다리 26 to 28
        video_right_down_leg = np.append(video_right_down_leg, np.array([[0,0]]), axis=0)#오른쪽 아래다리 25 to 27
    """

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
        cam_shoulder = np.append(cam_shoulder, np.array([poseData[11]-poseData[12]]), axis=0)# 양쪽 어깨 11 to 12
        cam_left_top_arm = np.append(cam_left_top_arm, np.array([poseData[12]-poseData[14]]), axis=0)# 왼쪽 윗팔 12 to 14
        cam_right_top_arm = np.append(cam_right_top_arm, np.array([poseData[11]-poseData[13]]), axis=0)#오른쪽 윗팔 11 to 13 
        cam_left_down_arm = np.append(cam_left_down_arm, np.array([poseData[14]-poseData[16]]), axis=0)#왼쪽 아래팔 14 to 16
        cam_right_down_arm = np.append(cam_right_down_arm, np.array([poseData[13]-poseData[15]]), axis=0)#오른쪽 아래팔 13 to 15
        cam_left_body = np.append(cam_left_body, np.array([poseData[12]-poseData[24]]), axis=0)#왼쪽 옆구리 12 to 24 
        cam_right_body = np.append(cam_right_body, np.array([poseData[11]-poseData[23]]), axis=0)#오른쪽 옆구리 11 to 23
        cam_hip = np.append(cam_hip, np.array([poseData[23]-poseData[24]]), axis=0)#양쪽 골반 23 to 24
        cam_left_top_leg = np.append(cam_left_top_leg, np.array([poseData[24]-poseData[26]]), axis=0)#왼쪽 윗다리 24 to 26 
        cam_right_top_leg = np.append(cam_right_top_leg, np.array([poseData[23]-poseData[25]]), axis=0)#오른쪽 윗다리 23 to 25
        cam_left_down_leg = np.append(cam_left_down_leg, np.array([poseData[26]-poseData[28]]), axis=0)#왼쪽 아래다리 26 to 28
        cam_right_down_leg = np.append(cam_right_down_leg, np.array([poseData[25]-poseData[27]]), axis=0)#오른쪽 아래다리 25 to 27



        poseData.clear()   
                
    #예외처리(참여자) 검출되지 않을 경우 0,0       
    except (TypeError, AttributeError):
        print(id)
    """
        cam_shoulder = np.append(cam_shoulder, np.array([[0,0]]), axis=0)# 양쪽 어깨 11 to 12
        cam_left_top_arm = np.append(cam_left_top_arm, np.array([[0,0]]), axis=0)# 왼쪽 윗팔 12 to 14
        cam_right_top_arm = np.append(cam_right_top_arm, np.array([[0,0]]), axis=0)#오른쪽 윗팔 11 to 13 
        cam_left_down_arm = np.append(cam_left_down_arm, np.array([[0,0]]), axis=0)#왼쪽 아래팔 14 to 16
        cam_right_down_arm = np.append(cam_right_down_arm, np.array([[0,0]]), axis=0)#오른쪽 아래팔 13 to 15
        cam_left_body = np.append(cam_left_body, np.array([[0,0]]), axis=0)#왼쪽 옆구리 12 to 24 
        cam_right_body = np.append(cam_right_body, np.array([[0,0]]), axis=0)#오른쪽 옆구리 11 to 23
        cam_hip = np.append(cam_hip,np.array([[0,0]]), axis=0)#양쪽 골반 23 to 24
        cam_left_top_leg = np.append(cam_left_top_leg, np.array([[0,0]]), axis=0)#왼쪽 윗다리 24 to 26 
        cam_right_top_leg = np.append(cam_right_top_leg, np.array([[0,0]]), axis=0)#오른쪽 윗다리 23 to 25
        cam_left_down_leg = np.append(cam_left_down_leg, np.array([[0,0]]), axis=0)#왼쪽 아래다리 26 to 28
        cam_right_down_leg = np.append(cam_right_down_leg, np.array([[0,0]]), axis=0)#오른쪽 아래다리 25 to 27
    """
    
path = dtw_ndim.warping_path(cam_right_down_leg, video_right_down_leg)      
print(path) 
"""
   
#유사도 비교 (임시)       
for i in range(len(video_shoulder)):
    print(i,"번째 프레임")
    sum=0
    if video_shoulder[i][0] == 0 or cam_shoulder[i][0] ==0:
        print("검출되지 않음")
    else:
        
        #양쪽 어깨 
        sim_score = score(video_shoulder[i],cam_shoulder[i])
        print("양쪽 어깨 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 윗팔
        sim_score = score(video_left_top_arm[i],cam_left_top_arm[i])
        print("왼쪽 윗팔 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 윗팔
        sim_score = score(video_right_top_arm[i],cam_right_top_arm[i])
        print("오른쪽 윗팔 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 아래팔
        sim_score = score(video_left_down_arm[i],cam_left_down_arm[i])
        print("왼쪽 아래팔 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 아래팔
        sim_score = score(video_right_down_arm[i],cam_right_down_arm[i])
        print("오른쪽 아래팟 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 옆구리
        sim_score = score(video_left_body[i],cam_left_body[i])
        print("왼쪽 옆구리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 옆구리
        sim_score = score(video_right_body[i],cam_right_body[i])
        print("오른쪽 옆구리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #양쪽 골반
        sim_score = score(video_hip[i],cam_hip[i])
        print("양쪽 골반 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 윗다리
        sim_score = score(video_left_top_leg[i],cam_left_top_leg[i])
        print("왼쪽 윗다리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 윗다리
        sim_score = score(video_right_top_leg[i],cam_right_top_leg[i])
        print("오른쪽 윗다리 : ",sim_score,"%")
        sum +=sim_score
        
        #왼쪽 아래다리
        sim_score = score(video_left_down_leg[i],cam_left_down_leg[i])
        print("왼쪽 아래 다리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        #오른쪽 아래다리
        sim_score = score(video_right_down_leg[i],cam_right_down_leg[i])
        print("오른쪽 아래다리 유사도 : ",sim_score,"%")
        sum +=sim_score
        
        print("평균 점수 : " ,sum/12,"\n")
"""    
print("종료")
cv2.destroyAllWindows()    
