
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

import math

path_to_bag = "8.bag"  # Location of input file.

config = rs.config()
# This specifies that we are loading pre-recorded data 
# rather than using a live camera.
config.enable_device_from_file(path_to_bag)
pipeline = rs.pipeline()  # Create a pipeline
profile = pipeline.start(config)  # Start streaming
align = rs.align(rs.stream.color)  # Create the alignment object
    

def draw_lines(lines, colour_set, base_image):
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(base_image,(x1,y1),(x2,y2),colour_set,2)    


# depth image processing params
sobel_kernel_size = 3
morph_size = 10
depth_canny_threshold1 = 3
depth_canny_threshold2 = 9
# RGB processing params
rgb_canny_threshold1 = 60
rgb_canny_threshold2 = 90

while True:
    
    # Get frameset of color and depth and align them.
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get the frames
    depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())
    color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
    

    #------------------------------ DEPTH ------------------------------

    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_RAINBOW)
    
    # use closing morphology on depth image to excentuate features
    morph_elem = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    
    # Dilate the image to make all features bigger, giving the canny edge detector an easier time
    dilation = cv2.dilate(closing,kernel,iterations = 1)

    # Use canny edge detector to find all depth changes in the image
    canny_depth = cv2.Canny(dilation, depth_canny_threshold1, depth_canny_threshold2)
    #cv2.imshow("Depth_canny", canny_depth)
    
    minLineLength_depth = 0
    maxLineGap_depth = 600
    
    #throw hough at depth
    lines_depth = cv2.HoughLinesP(canny_depth, 1, np.pi/180, 100, minLineLength=minLineLength_depth, maxLineGap=maxLineGap_depth)   
    
    if lines_depth is not None:
        if len(lines_depth) > 0:
            draw_lines(lines_depth, (0,255,0), dilation)  
        cv2.imshow("Hough_Depth", dilation)    


    #------------------------------- RGB -------------------------------
    
    #throw a canny at RGB
    gray = cv2.Canny(color_image, rgb_canny_threshold1, rgb_canny_threshold2)
    #cv2.imshow("Canny", gray)    
    
    #throw hough at RGB
    minLineLength = 1
    maxLineGap = 850
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

    # only keep lines that are vertical 
    lines_filtered = []
    
    # Removing all lines that arenot vertical
    # Angle limits for each set
    gradient_h_max = 5
    gradient_h_min = -5
    
    gradient_v_max = 100
    gradient_v_min = 80
    
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                
                gradient = abs(y1-y2) / abs(x1-x2)
                angle = math.degrees(math.atan(gradient))
                
                #if (angle > gradient_h_min) and (angle < gradient_h_max):
                    #lines_filtered.append(line)
                if (angle > gradient_v_min) and (angle < gradient_v_max):
                    lines_filtered.append(line)

    
    # Draw a set of lines on image
    
    #if len(lines_filtered) > 0:
        #draw_lines(lines_filtered, (0,255,0), color_image)  
    #cv2.imshow("Hough", color_image)
    
    # ---------------------- Doorway Detection ----------------------
    
    sparation_threshold_pixels = 10
    
    intersections = []
    
    if lines_filtered is not None and lines_depth is not None:
        for line_depth in lines_depth:  
            
            for x1_d, y1_d, x2_d, y2_d in line_depth:
                gradient_d = abs(y1_d-y2_d) / abs(x1_d-x2_d)
                angle_d = math.degrees(math.atan(gradient_d))  
                
            for line_rgb in lines_filtered:
                
                for x1_rgb, y1_rgb, x2_rgb, y2_rgb in line_rgb:
                    gradient_rgb = abs(y1_rgb-y2_rgb) / abs(x1_rgb-x2_rgb)
                    angle_rgb = math.degrees(math.atan(gradient_rgb))
                    
                [x1_d, x2_d] = sorted([x1_d, x2_d])
                [y1_rgb, y2_rgb] = sorted([y1_rgb, y2_rgb])   
                sparation = abs(x1_rgb - x1_d)
                
                if sparation < sparation_threshold_pixels:
                    intersections.append((line_rgb, sparation))
                    
                
    if len(intersections) > 0:
        for line, separation in intersections:
            
            hit_percentage = (sparation_threshold_pixels - separation) / 10
            print(hit_percentage)
            
            for x1,y1,x2,y2 in line:
                cv2.line(color_image,(x1,y1),(x2,y2),(0,255,0),2) 
                
                color_image = cv2.putText(color_image, str(hit_percentage), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
                
    cv2.imshow("Hough", color_image)    
    
    
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Close the script when q is pressed.
        break
    

#cap.release()
cv2.destroyAllWindows()
