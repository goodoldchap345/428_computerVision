# d435_from_file.py

import pyrealsense2 as rs
import numpy as np
import cv2

path_to_bag = "night.bag"  # Location of input file.

config = rs.config()
# This specifies that we are loading pre-recorded data 
# rather than using a live camera.
config.enable_device_from_file(path_to_bag)

pipeline = rs.pipeline()  # Create a pipeline
profile = pipeline.start(config)  # Start streaming
align = rs.align(rs.stream.color)  # Create the alignment object

while True:
    # Get frameset of color and depth and align them.
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get the frames
    depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())
    color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())

    # Show the depth and color data to the screen.
    cv2.imshow('Colour', color_image)
    cv2.imshow('Depth', depth_image)
    
    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and close the GUI windows.
pipeline.stop()
cv2.destroyAllWindows()
