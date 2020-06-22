
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

import math

# Path to input file (.bag)
path_to_bag = "2.bag" 

config = rs.config()
# This specifies that we are loading pre-recorded data 
# rather than using a live camera.
config.enable_device_from_file(path_to_bag)
pipeline = rs.pipeline()  # Create a pipeline

try:
    profile = pipeline.start(config)  # Start streaming
except RuntimeError:
    print("File not found")
    sys.exit()
    
align = rs.align(rs.stream.color)  # Create the alignment object
    

def draw_lines(lines, colour_set, base_image):
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(base_image,(x1,y1),(x2,y2),colour_set,2)    


while True:
    
    # Align RGB and depth images 
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Extract depth and RGB images from aligned video stream
    depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())
    color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
    

    #------------------------------ DEPTH ------------------------------

    # Apply colour to depth image to get it to work with canny edge detector
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_RAINBOW)
    
    
    morph_size = 10
    
    # Form morphology kernel 
    morph_elem = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    
    # Use closing morphology on depth image to excentuate features
    closing = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    
    # Dilate the image to make all features bigger, giving the canny edge detector an easier time
    dilation = cv2.dilate(closing,kernel,iterations = 1)


    depth_canny_threshold1 = 3
    depth_canny_threshold2 = 9    

    # Use canny edge detector to find all depth changes in the image
    canny_depth = cv2.Canny(dilation, depth_canny_threshold1, depth_canny_threshold2)
    
    
    minLineLength_depth = 0
    maxLineGap_depth = 600
    
    # Use a probabilistic hough transform to find all lines in the depth image
    lines_depth = cv2.HoughLinesP(canny_depth, 1, np.pi/180, 100, minLineLength=minLineLength_depth, maxLineGap=maxLineGap_depth)   
    
    if lines_depth is not None:
        if len(lines_depth) > 0:
            draw_lines(lines_depth, (0,255,0), dilation)  
        cv2.imshow("Hough_Depth", dilation)    


    #------------------------------- RGB -------------------------------
    
    rgb_canny_threshold1 = 60
    rgb_canny_threshold2 = 90    
    
    # Use a canny edge detector on the RGB image
    gray = cv2.Canny(color_image, rgb_canny_threshold1, rgb_canny_threshold2)    

    
    minLineLength_RGB = 1
    maxLineGap_RGB = 850    
    
    # Apply a hough transform to the RGB image
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100,
                            minLineLength=minLineLength_RGB, maxLineGap=maxLineGap_RGB)


    gradient_v_max = 100
    gradient_v_min = 80    

    # only keep lines that are vertical 
    lines_filtered = []
    
    # If a line has a gradient not within the set thresholds, remove it
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                
                gradient = abs(y1-y2) / abs(x1-x2)
                angle = math.degrees(math.atan(gradient))
                
                if (angle > gradient_v_min) and (angle < gradient_v_max):
                    lines_filtered.append(line)

    
    # ---------------------- Doorway Detection ----------------------
    
    separation_threshold_pixels = 10
    
    intersections = []
    
    # Iterate through all RGB lines, work out seperation from nearest depth image line
    if lines_filtered is not None and lines_depth is not None:
        for line_rgb in lines_filtered:
            intersection_found = False
            # Calculate the gradient
            for x1_rgb, y1_rgb, x2_rgb, y2_rgb in line_rgb:
                gradient_rgb = abs(y1_rgb-y2_rgb) / abs(x1_rgb-x2_rgb)
                angle_rgb = math.degrees(math.atan(gradient_rgb))            
                
            for line_depth in lines_depth: 
                # Calculate the gradient
                for x1_d, y1_d, x2_d, y2_d in line_depth:
                    gradient_d = abs(y1_d-y2_d) / abs(x1_d-x2_d)
                    angle_d = math.degrees(math.atan(gradient_d))                  
                    
                [x1_d, x2_d] = sorted([x1_d, x2_d])
                [y1_rgb, y2_rgb] = sorted([y1_rgb, y2_rgb])   
                # Find the separation between the two vertical lines
                separation = abs(x1_rgb - x1_d)
                
                # If the RGB line is further from the depth line than the threshold
                if separation < separation_threshold_pixels:
                    # Update or append the separation value
                    if not intersection_found:
                        intersections.append((line_rgb, separation))
                    elif separation < intersections[-1][1]:
                        intersections[-1][1] = separation
                    
    
    # Overlay all doorway detections onto the base colour image
    if len(intersections) > 0:
        for line, separation in intersections:
            # Calculate the probability that these lines do correspond to a door
            door_probability = (separation_threshold_pixels - separation) / 10
            
            for x1,y1,x2,y2 in line:
                cv2.line(color_image,(x1,y1),(x2,y2),(0,255,0),2) 
                # Write the probability of the line being a doorway onto the image
                color_image = cv2.putText(color_image, str(door_probability), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
                
    cv2.imshow("Doorway detections", color_image)    
    
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Close the script when q is pressed.
        break
    

#cap.release()
cv2.destroyAllWindows()
