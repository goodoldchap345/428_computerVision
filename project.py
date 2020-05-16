# harris.py

import cv2
import numpy as np
#import pyrealsense2 as rs

cap = cv2.VideoCapture('doorway.avi')  # Open the first camera connected to the computer.

while True:
    ret, frame_raw = cap.read()
    frame = frame_raw[800:1200, 200:700]

    #throw a canny at it
    threshold1 = 75
    threshold2 = 200
    gray = cv2.Canny(frame, threshold1, threshold2)

    cv2.imshow("Canny", gray)

    #through hough at it
    minLineLength = 2
    maxLineGap = 500
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("Hough", frame)
    
    # The Harris corner detector operates on a grayscale image.
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(gray,2,3,0.04)
    # Dialate the detected corners to make them clearer in the output image.
    corners = cv2.dilate(corners,None)

    # Perform thresholding on the corners to throw away some false positives.
    frame[corners > 0.02 * corners.max()] = [0,0,255]

    cv2.imshow("Harris", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):  # Close the script when q is pressed.
        break

cap.release()
cv2.destroyAllWindows()
