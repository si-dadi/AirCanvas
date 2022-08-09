
#         AirCanvas - Draw using your Hands!

#                   Drawing Modes:
#   (Hold the below mentioned fingers near each other)
# Touch the Tips of Thumb and Index Fingers: Selection Mode
#               Index Up: Drawing Mode
#           Index and Middle Up: Hold Mode


from sre_constants import SUCCESS
from turtle import fillcolor, width
import cv2
import numpy as np, os
import mediapipe as mp

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image= cv2.imread(f'{folderPath}\{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)     

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.85,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Starting the painting window setup
paintWindow = np.zeros((471,636,3)) + 255

colors = [(0,0,0), (22,22,225), (158,152,3), (55,128,0), (77,145,255), (235,23,94)]
colorIndex = 0
drawColor = colors[0]
drawThickness = 2
xp,yp = 0,0

while True:

    # Import Image and flip
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Find hand Landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Set drawing Modes:
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = img.shape

            x1,y1 = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x*w , hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y*h  
            x2,y2 = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x*w , hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y*h
            x3,y3 = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x*w , hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y*h

            # Drawing Modes:    Thumb and Index Up: Selection Mode  Index Up: Drawing Mode  Index and Middle Up: Hold Mode
            
            #Selection Mode:
            if ( y1 < 62 and y3 < 62):
                x,y = (x1+x3)/2, (y1+y3)/2

                if 75 < x < 170:
                    drawColor = colors[0]
                    drawThickness = 2
                    # print('Black')

                if 175 < x < 225:
                    drawColor = colors[1]
                    drawThickness = 2
                    # print('Magenta')

                if 230 < x < 300:
                    drawColor = colors[2]
                    drawThickness = 2
                    # print('Cyan')

                if 310 < x < 390:
                    drawColor = colors[3]
                    drawThickness = 2
                    # print('Green')

                if 400 < x < 475:
                    drawColor = colors[4]
                    drawThickness = 2
                    # print('Orange')

                if 480 < x < 535:
                    drawColor = colors[5]
                    drawThickness = 2
                    # print('Violet')

                if 540 < x < 630:
                    drawColor = (255,255,255)
                    drawThickness = 25
                    # print('Eraser)


            # Drawing Mode:
            if(abs(y1-y2) >20 and abs(y1-y3) > 20):
                cv2.circle(img, (int(x1), int(y1)), 10, (255,255,0), -1) 

                if (xp == 0 and yp == 0):
                    xp,yp = x1,y1
                
                cv2.line(img, (int(xp),int(yp)), (int(x1),int(y1)), drawColor, drawThickness) 
                cv2.line(paintWindow, (int(xp),int(yp)), (int(x1),int(y1)), drawColor, drawThickness)
                xp,yp = x1,y1 


            # Hold Mode:
            if((abs(x1-x2) < 20) and (abs(y1-y2) < 20)):
                cv2.circle(img, (int((x1+x2)/2), int((y1+y1)/2)), 15, (0,0,255), -1)
            xp,yp = x1,y1
            

    # Setting Header Image
    img[0:62, 0:640] = header
    cv2.imshow("Live Feed", img)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows