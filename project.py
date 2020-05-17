# harris.py

import sys
import cv2
import numpy as np
import pyrealsense2 as rs

# ------- For standard video file --------
#cap = cv2.VideoCapture('doorway.avi')  # Open the first camera connected to the computer.
#if cap.read()[1] is None:
    #print("file not found")
    #sys.exit()
# ----------------------------------------


path_to_bag = "3.bag"  # Location of input file.

config = rs.config()
# This specifies that we are loading pre-recorded data 
# rather than using a live camera.
config.enable_device_from_file(path_to_bag)
pipeline = rs.pipeline()  # Create a pipeline
profile = pipeline.start(config)  # Start streaming
align = rs.align(rs.stream.color)  # Create the alignment object
    

while True:
    #ret, frame_raw = cap.read()
    #frame = frame_raw[800:1200, 200:700]
    
    # Get frameset of color and depth and align them.
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get the frames
    depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())
    color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
    frame = color_image

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
    print(lines)
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

#cap.release()
cv2.destroyAllWindows()
