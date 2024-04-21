import cv2 as cv
import mediapipe as mp
import numpy as np
from cvzone.ClassificationModule import Classifier
import tensorflow
import time
import os




# video = cv.VideoCapture(0)
labels = ["Mukha Svanasana","Utkata Konasana","Falasana","Vrikshasana","Virabhadrasana"]
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
classifier = Classifier("/home/ujjawal/Python/AdvanceComputerVision2/Model/keras_model.h5","/home/ujjawal/Python/AdvanceComputerVision2/Model/labels.txt")

pTime = 0
cTime = 0
lmlist =[]
lmlist_x = []
cxleft = 0
cxright = 0
cytop =0 
offset = 20
counter=0



    



img = cv.imread("/home/ujjawal/Python/AdvanceComputerVision2/DATASET/TEST/plank/00000008.jpg")
# cv.imshow("IMAGE",img)

h1,w1,_ = img.shape
img_crop = np.ones((300,300,3),np.uint8)*255

imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
results = pose.process(imgRGB)
imgWhite = np.ones((300,300,3),np.uint8)*255
# print(results.pose_landmarks)
img2 = np.ones((h1,w1,3),np.uint8)
# counter +=1

if results.pose_landmarks:
    mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    mpDraw.draw_landmarks(img2,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    lmlist = []
    lmlist_x = []
    lmlist_y = []
    #   for handlms in results.pose_landmarkss
    for id , lm in enumerate(results.pose_landmarks.landmark):
                
        #     print(id,lm)
            h,w,c =img.shape
            cx,cy = int(lm.x*w) , int(lm.y*h)
        #     print(id,cx,cy)
            lmlist.append([id,cx,cy])
            
            lmlist_x.append(cx)
            lmlist_y.append(cy)

cxright = min(lmlist_x)
cxleft = max(lmlist_x)
cytop = min(lmlist_y)
cybottom = max(lmlist_y)
if cxright - offset >= 0 and cytop - offset >= 0 and cxleft + offset <= w1 and cybottom + offset <= h1:
    cv.rectangle(img , (cxright - offset,cytop - offset), (cxleft + offset,cybottom +  offset) , (0,255,0),2)


    img_crop = img[cytop - offset:cybottom +  offset , cxright - offset:cxleft + offset]
    img_crop = cv.resize(img_crop, (500,500), interpolation= cv.INTER_AREA)


    img_crop2 = img2[cytop - offset:cybottom +  offset , cxright - offset:cxleft + offset]
    img_crop2 = cv.resize(img_crop2, (500,500), interpolation= cv.INTER_AREA)


    img_crop_shape = img_crop.shape

    prediction,index = classifier.getPrediction(img_crop2)
    print(prediction , index)
  
    cv.rectangle(img,(cxright - offset , cytop + offset -80) , (cxright - offset +50 , cytop + offset -38),(0,255,0),cv.FILLED)
    cv.putText(img ,labels[index] ,(cxright-12,cytop-23),cv.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),1)


# imgWhite[0:img_crop_shape[0],0:img_crop_shape[1]]= img_crop


                
            

            


    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img , str(int(fps)),(10,100),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                
    cv.imshow('IMG',img)
    # cv.imshow('crop',img_crop)
    # cv.imshow('FINAL',img_crop2)
    # cv.imshow('new',img2)
    # cv.imshow('newcrop',img_crop2)
    # cv.imshow('white',imgWhite)

    cv.waitKey(0)
 


        
        
       

    

    # if key == ord('q'):
    #     break
    
    
    # counter +=1
    # cv.imwrite(f'{folder}/Image_{time.time()}.jpg',img_crop2)
    # print(counter)
            

cv.destroyAllWindows()