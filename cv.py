
# Note TO self never use OPENCV(PYTHON) in jupyter notebook it will crash the kernel on a mac 
# Use a python script instead

# Import the necessary libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2.aruco as aruco 
import liqdist_archit as liqdist
import yaml



# Load the vid 
vid = cv2.VideoCapture('Vids/2.8 bar.mp4')

# Load the aruco 
parameters = aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

def show_vid(vid, frame_rate=30):
    # Calculate the delay based on the frame rate
    delay = int(1000 / frame_rate)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)
        
        # Use the calculated delay in cv2.waitKey()
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print('Quitting...')
            break

    
    vid.release()
    cv2.destroyAllWindows()
    

# Load the camera matrix and distortion coefficients from the YAML file
with open('cal_mat.yaml') as f:
    data = yaml.safe_load(f)
    K = np.array(data['camera_matrix'])
    D = np.array(data['dist_coeff'])

# Print the sizes of K and D for debugging
print('K:', K)
print('K.shape:', K.shape)
print('D:', D)
print('D.shape:', D.shape)

def detect_aruco(vid, dictionary, parameters):
    max_area = 0
    max_frame = None

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # If 4 markers are detected, draw a rectangle
        if ids is not None and len(ids) == 4:
            # Compute the center of each marker
            centers = [np.mean(corner, axis=1)[0] for corner in corners]
            # Compute the centroid of all markers
            centroid = np.mean(centers, axis=0)
            # Compute the angle of each marker relative to the centroid
            angles = [np.arctan2(center[1] - centroid[1], center[0] - centroid[0]) for center in centers]
            # Sort the centers by their angles
            centers = [center for _, center in sorted(zip(angles, centers))]
            # Add the first center to the end to close the loop
            centers.append(centers[0])
            # Draw lines between the centers
            for i in range(len(centers) - 1):
                pt1 = tuple(np.intp(centers[i]))
                pt2 = tuple(np.intp(centers[i + 1]))
                frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            # Compute the area using the Shoelace formula
            area = 0.5 * abs(sum(x1*y2 - x2*y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))
            # If the area is the maximum so far, save the frame
            if area > max_area:
                max_area = area
                max_frame = frame.copy()

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quitting...')
            break

    # Save the frame with the maximum area
    if max_frame is not None:
        cv2.imwrite('max_area_frame.png', max_frame)

    vid.release()
    cv2.destroyAllWindows()

detect_aruco(vid, dictionary, parameters)
    







