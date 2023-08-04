from sre_constants import SUCCESS
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/1.mp4')#동영상
pTime = 0
while True:
    SUCCESS, img = cap.read()#이미지를 한프레임씩 읽어옴
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks) #좌표 출력
    if results.pose_landmarks:#랜드마크가 있다면
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c,=img.shape
            print(id,lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
    
    
    cTime = time.time()
    fps = 1 /(cTime-pTime)
    pTime = cTime
    #프레임 출력
    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    #이미지 출력
    cv2.imshow("Image",img)
    cv2.waitKey(1)