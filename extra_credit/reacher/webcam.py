#uses opencv: https://pypi.org/project/opencv-python/
#computer vision (webcam stuff)
#RBT350 -- extra credit
#Gabriel Moore, Luke Pronga, Maadhav Kothuri

import cv2

#   ---                                               ---    #
#   ---   CHANGE THIS TO 0 IF YOU'RE NOT DEBUGGING    ---    #
#   ---                                               ---    #
debug = 1
viz = 1

#use laptop webcam
cap = cv2.VideoCapture(0)



def webcam():
    #read webcam frame
    ret, frame = cap.read()
    red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bottom = (0, 50, 50)
    top = (10, 255, 255)
    mask = cv2.inRange(red, bottom, top)   #define range of reds to find in frame

    #find contours of the red regions in the mask
    contours, buttstuff = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    contours.sort(key=cv2.contourArea, reverse=True)

    #take only the largest contour
    if len(contours) > 0:
        largest_contour = contours[0]
        # Draw a circle around the largest contour
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        cv2.circle(frame, (int(x + w/2), int(y + h/2)), 10, (0, 0, 255), 2)
    else:
        x = -1
        y = -1
    contours, buttstuff = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return x,y,frame



def find_dot():
    if(debug == 0):
        #y = left-right on robot
        #z = up-down on robot
        x,y,frame = webcam()
        if(viz == 1):   #display visual data
            cv2.putText(frame, "x:" + str(x), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(frame, "y:" + str(y), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow('Red Dot Detection', frame)
        #scale to match robot (w x h = 640 x 480)
        #translate to range of -0.2 to 0.2
        return x,y
    else:
        while debug:
            #     --DEBUG ONLY--      #
            x,y,frame = webcam()
            cv2.putText(frame, "x:" + str(x), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(frame, "y:" + str(y), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow('Red Dot Detection', frame)
            if(x > -1): print("x:",x,"  y:",y)

            #exit the loop if escape key is pressed
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
    return



find_dot()