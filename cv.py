
# Note TO self never use OPENCV(PYTHON) in jupyter notebook it will crash the kernel on a mac 
# Use a python script instead

# Import the necessary libraries
import cv2 
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import os
import cv2.aruco as aruco 
import liqdist_archit as liqdist
import yaml
import time


filename = "OneShot_1.mp4"
# Load the vid 
vid = cv2.VideoCapture(f'Vids/{filename}')
# Load the aruco 
parameters = aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)




# def show_vid(vid, frame_rate=30):
#     # Calculate the delay based on the frame rate
#     delay = int(1000 / frame_rate)

#     while vid.isOpened():
#         ret, frame = vid.read()
#         if not ret:
#             break
#         cv2.imshow('Frame', frame)
        
#         # Use the calculated delay in cv2.waitKey()
#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             print('Quitting...')
#             break

    
#     vid.release()
#     cv2.destroyAllWindows()
    
        



# def detect_aruco(vid, dictionary, parameters):
#     max_area = 0
#     max_frame = None
#     offset_y = 10  # Define the offset for y
#     count = 0 # Initialize the count to 0
#     cam_mat , dist_coeff = liqdist.read_cam_calibration()
#     frame_counter = 0  # Initialize a frame counter
#     cooldown_frames = 30*8 # Set the number of frames to skip after capturing a frame

#     while vid.isOpened():
#         ret, frame = vid.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         skewness = skew(gray.ravel())

#         corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
#         #frame = liqdist.intrinsic(frame, cam_mat, dist_coeff) 

#         # If 4 markers are detected, compute the area
#         if ids is not None and len(ids) == 4:
#             # Compute the center of each marker
#             centers = [np.mean(corner, axis=1)[0] for corner in corners]
#             # Compute the centroid of all markers
#             centroid = np.mean(centers, axis=0)
#             # Compute the angle of each marker relative to the centroid
#             angles = [np.arctan2(center[1] - centroid[1], center[0] - centroid[0]) for center in centers]
#             # Sort the centers by their angles
#             centers = [center for _, center in sorted(zip(angles, centers))]
#             # Add the first center to the end to close the loop
#             centers.append(centers[0])
#             # Compute the area using the Shoelace formula
#             area = 0.5 * abs(sum(x1*y2 - x2*y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))
#             # If the area is the maximum so far, save the frame and output the timestamp
            
#             if area > max_area  and frame_counter >= cooldown_frames :
#                 max_area = area
#                 max_frame = frame.copy()
#                 timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)  # Get the timestamp of the current frame
#                 print(f"New max area found at {round((timestamp/1000)/60,1)} min {round((timestamp/1000) % 60,1)} s and the area is {area} pixels") 

#                 # Specify the directory name
#                 dir_name = "output_frames"

#                 # Check if the directory exists
#                 if not os.path.exists(dir_name):
#                     # If the directory does not exist, create it
#                     os.makedirs(dir_name)

#                 # Save the frame with the maximum area after the video has been processed
#                 if max_frame is not None:
#                     cv2.imwrite(f'output_frames/max_area_frame_{os.path.splitext(filename)[0]}_{count}.png', max_frame)
#                     max_frame = None # Reset the max_frame variable
#                     count += 1  # Increment the count
#                     frame_counter = 0  # Reset the frame counter

#         frame_counter += 1  # Increment the frame counter for each frame

#     vid.release()
#     cv2.destroyAllWindows()

#detect_aruco(vid, dictionary, parameters)

liqdist.DEBUG_MODE = True
# Get a list of all the files in the output frames folder
output_frames_folder = 'output_frames'
filenames = os.listdir(output_frames_folder)

# Crop the frames of the arucos and save them in a new folder
for filename in filenames:
    # Read the image
    img = cv2.imread(f'{output_frames_folder}/{filename}')
    # Crop the image
    cropped_img = liqdist.crop_image(img)
    # Save the cropped image
    cv2.imwrite(f'cropped_frames/{filename}', cropped_img)





