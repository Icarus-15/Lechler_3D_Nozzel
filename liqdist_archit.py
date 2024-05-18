# Author: Archit Gupta

#### SYSTEM MODES ####
DEBUG_MODE = False
PRODUCTION_MODE = True

#############################################################################
# Libraries ....
#############################################################################
import cv2
from cv2 import aruco
import os
import yaml
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

#Additionally required: camera calibration file
################################################### ##########################
#Read camera calibration file

def read_cam_calibration(): 
    with open("cal_mat.yaml", "r") as f:
        read_data = yaml.load(f, Loader=yaml.FullLoader)
        camera_matrix = np.array(read_data['camera_matrix'])
        distortion_coefficients = np.array(read_data['dist_coeff'])
        return(camera_matrix, distortion_coefficients)

################################################### ##########################
#Selecting the IP camera (cameras should be labeled)
#Untested from my side 
def select_ipcamera():
    selected_cap = None  # Initialize selected_cap with a default value
    if PRODUCTION_MODE:
        print('Select image source: ')
        print('_____________________________')
        print('(1)...Camera IP 10.130.191.134')
        print('(2)...Camera IP 10.49.235.169')
        print('(3)...Camera IP 10.49.235.171')
        print('(4)...Camera IP 10.49.235.46')
        print('(5)...Open image from file')
        print('(6)...iPhone guko')
        
        selector = input('Select Source(1-6): ')
        if selector == '1': 
            selected_cap = cv2.VideoCapture('rtsp://admin:lechler@123@10.130.191.134/Streaming/channels/2')
        elif selector == '2': 
            selected_cap = cv2.VideoCapture('rtsp://admin:L3chl3rGmbH@10.49.235.169:80')
        elif selector == '3': 
            selected_cap = cv2.VideoCapture('rtsp://admin:L3chl3rGmbH@10.49.235.171:80')
        elif selector == '4':
            selected_cap = cv2.VideoCapture('rtsp://admin:LechlerREA@10.49.235.46:80')
        elif selector == '5':
            image_path = "./test_image.jpg"
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(initialdir='C:\temp')
            # selected_cap = cv2.imread(file_path)
            print("Image Path:", file_path)  # Debugging print
            selected_cap = cv2.imread(file_path)
            if selected_cap is None:
                print('Error loading image')
            else:
                print('Image loaded successfully')  # Debugging print
                pass

        elif selector == '6':
            pass  # Placeholder for future code

        else:
            print('Invalid Selection')
            selected_cap = 0
    elif DEBUG_MODE:
        selector = '5'
        print("Importing image as a way to debug")
        ##FOR_DEBUG: Specify the image path
        image_path = "./7.jpg"
        print("Image Path entered: ",image_path)
        selected_cap = cv2.imread(image_path)
    else:
        print("Check System Settings")
    return selected_cap, str(selector)

################################################### ##########################
#Track and display aruco markers in live images.
#Source: German Code writer, Not tested by me
def track(cap,matrix_coefficients, distortion_coefficients):
    print("j")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)  # Load dictionary for detection
    parameters = cv2.aruco.DetectorParameters()  # Generate parameters for detection of the markers
    key = 0 #initialize key before loop
    cv2.namedWindow('Search...', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        print("Capture Sucess", ret) #For debugging 
        if not ret:
            print("Error Capturing frame from camera")
            break
        frame_copy = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Convert to grayscale for detection of Aruco markers
        
        #List of corners, IDs and rejected results as return of the detection        
        corners, ids, rejected = aruco.detectMarkers(frame_copy, aruco_dict,
                                                    parameters=parameters)
        arucofound = detect_arucos(frame_copy, matrix_coefficients, distortion_coefficients)
        if arucofound is not None:
         if np.all(ids is not None):
            
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.01, matrix_coefficients, distortion_coefficients)
                aruco.drawDetectedMarkers(frame_copy, corners)
                length = 0.01  # Length of the axis lines
                axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
                image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, matrix_coefficients, distortion_coefficients)
    
                origin = tuple(corners[i][0][0])  # Marker corner point
                x_end = tuple(image_points[1].ravel())
                y_end = tuple(image_points[2].ravel())
                z_end = tuple(image_points[3].ravel())
    
                cv2.line(frame_copy, np.int32(origin), np.int32(x_end), (255, 0, 0), 2)  # Draw X-axis in blue
                cv2.line(frame_copy, np.int32(origin), np.int32(y_end), (0, 255, 0), 2)  # Draw Y-axis in green
                cv2.line(frame_copy, np.int32(origin), np.int32(z_end), (0, 0, 255), 2)  # Draw Z-axis in red
        cv2.imshow('Search...', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return frame_copy

# Modified: Added Fisheye Model based undistortion
def intrinsic(image, camera_matrix, distortion_coefficients):
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(image, camera_matrix, distortion_coefficients, None, newcameramtx)
    # map1,map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix,distortion_coefficients,np.eye(3),newcameramtx,size=(w,h),m1type=cv2.CV_16SC2)
    # undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return (undistorted_img)




################################################### ####################
#Find Aruco markers in image, sort and return center points #
def detect_arucos(image, camera_matrix, distortion_coefficients):
    gray = image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Create a parameters dictionary and set values manually
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMethod = 0
    parameters.markerBorderBits = 1
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    parameters.maxErroneousBitsInBorderRate = 0.04
    parameters.minOtsuStdDev = 5.0
    parameters.errorCorrectionRate = 0.6

    detector =  cv2.aruco.ArucoDetector(aruco_dict, parameters)
    #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    print("Detected ArUco Markers:")
    print("IDs:", ids)
    print("Corners:", corners)
    
    if np.all(ids is not None):
        if len(ids) != 4:
            print('ArUcoMarker error. Find',len(ids),' of four.')
            return None
        elif len(ids) ==4:
            print('4 ArUco marker detected.')
            #Sort markers by identities in ascending order ---> otherwise subsequent assignment RL difficult +
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            corners = np.asarray(corners)
            ids = np.asarray(ids)
            # frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids)
            marker_centers = []
            for i in range(len(ids)):
                c = corners[i][0]
                d = [(c[0][0] + c[2][0]) / 2, (c[0][1] + c[2][1]) / 2]
                marker_centers.append(d)
            return marker_centers
    else: 
        print('No Aruico Marker Found please check camera position.')
        return None #return none instead of zero
##################################
#Perspective Transformation #
def crop_image(img, arucofound):
    marker_centers = arucofound
    # define size of new image in pixels
    rows = 2500
    cols = 5000
    
    # Source Points - coordinates of detected ArUcos 
    src_points = np.float32([
    marker_centers[0],  # Point 1
    marker_centers[1],  # Point 2
    marker_centers[2],  # Point 3
    marker_centers[3]   # Point 4
])

 
    # Destination Points - destination points for the transformation (= measured real coordinates in mm in the local system) +
    dst_1 = [1000,1000]
    dst_2 = [4695,1000]
    dst_3 = [1000,1445]
    dst_4 = [4695,1445]
    dst_points = np.float32([dst_1, dst_2, dst_3, dst_4]) #build... ascending order since Arucos are sorted!
    
    # Determine the transformation matrix with SourcePoints and DestinationPoints
    affine_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    #Perform perspective transformation +
     # cv.WarpPerspective(src, dst, mapMatrix, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray =img
    img_warp = cv2.warpPerspective(gray, affine_matrix, (cols,rows))
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Specify height and width for crop: Centers of the ArUco markers +
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    h = dst_3[1]-dst_1[1]
    w = dst_4[0]-dst_3[0]

    #Crop the image area so that only the measuring section can be seen.
    # img_crop = img_warp[dst_1[1]+200:dst_1[1]+h+110, dst_1[0]+260:dst_1[0]+w-400]
    img_crop = img_warp[dst_1[1] + int(h*0.01):dst_1[1]+h -int(h*0.01), dst_1[0] + int(w*0.03):dst_1[0]+w - int(w*0.03)]
    if DEBUG_MODE:
        print("Cropped image size",img_crop.shape)

    return (img_crop)

##############################
#Morphological operations #
def morphologic(img_crop):
    img_gray = img_crop
    #Convert image to grayscale and binary.
    #Set threshold for grayscale conversion!
    threshold = 200
    img_neg = 255-img_gray
    img_neg= img_neg + 0.2*cv2.Sobel(img_neg,cv2.CV_64F,0,1,ksize=5)
    ret, img_bw = cv2.threshold(img_neg,threshold,255,cv2.THRESH_BINARY)
    
    # Debug Mode
    if DEBUG_MODE:
        print("In function morphologic, there are a series of filters applied to extract a kind of a ball at the place of marker") 
    
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),iterations=3)
    img_bw = cv2.erode(img_bw,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
    img_bw = cv2.erode(img_bw,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)
    img_bw = cv2.erode(img_bw,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
    img_bw = cv2.dilate(img_bw,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=3)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)),iterations=1)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)),iterations=1)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9)),iterations=1)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=3)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_CROSS,(7,1)),iterations=2)
    img_bw = cv2.morphologyEx(img_bw,cv2.MORPH_DILATE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
    # # #morph -dilate:
    # # #Create kernel for second folding operation and execute dilate 
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    # img_bw = cv2.dilate(img_bw, kernel, iterations=1)
    img_bw = img_bw.astype(np.uint8)
    return (img_bw)

###################################
#Find balls and write file #

def find_columns(img_raw):
    # img_raw= img_raw+cv2.Sobel(img_raw,cv2.CV_8U,1,0,ksize=5)
    if DEBUG_MODE:
        print("This function gets columns as the tubes.")

    kernel = np.array([[0,-2,0],
                       [-2,9,-2],
                       [0,-2,0]])
    img_raw = cv2.filter2D(img_raw,-1,kernel=kernel)
    dst = cv2.Canny(img_raw, 60, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(dst, 1, 5*np.pi / 180, 0, None, 30, 5)
    
    if DEBUG_MODE:
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i]
                if (l[0][1]<30 and l[0][3]>30) or (l[0][1]>30 and l[0][3]<30) and l[0][2] == l[0][0]:
                    l=l[0]
                    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    return cdstP,linesP

def find_islands():
    
    pass

def find_balls(img_raw,img_undist):
    #++++++++++++++++++++++++++++++++
    #+ Find balls on the measuring track +
    #++++++++++++++++++++++++++++++++
    #We will use an averaging in the height instead of hough transform to find exact number of balls in the specific columns
    img_bw = img_raw.copy()
    #print(img_bw.shape)
    kernel = np.array([[0,-2,0],
                       [-2,9,-2],
                       [0,-2,0]])
    img_bw = cv2.filter2D(img_bw,-1,kernel=kernel)
    ##### DEUBG MODE ####
    if PRODUCTION_MODE:
        no_of_cylinders = 75 #int(input("Enter the expexcted number of Cylinders"))
        each_cylinder_width_mm = int(input("Enter the width of Cylinders in mm"))
        each_cylinder_height_cm = float(input("Enter the Height of Cylinders in cm"))
    if DEBUG_MODE:
        no_of_cylinders = 75
        each_cylinder_width_mm = 15
        each_cylinder_height_cm = 16.6
    
    each_cylinder_width_pix = img_bw.shape[1]//no_of_cylinders
    
    edged = cv2.Canny(img_bw, 20, 140)
    if DEBUG_MODE:
        print("Edges of the markers")
        plt.figure(figsize=(200,200))
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB))
        plt.axis('off')
        plt.show()
    
    # Calibration needed if new setup
    circles = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1.1, minDist=each_cylinder_width_pix*3//4, param1=5, param2=6.5, minRadius = 3, maxRadius = 15)  
    if DEBUG_MODE:
        print("MArkers in black and white")
        plt.figure(figsize=(200,200))
        plt.imshow(cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB))
        plt.axis('off')
        plt.show()
    
    if circles is not None: 
        #Convert to integer, otherwise it won't loop.
        circles = np.round(circles[0, :]).astype("int")
        #circles = np.round(circles[0, :]/25)*25.astype("int")
        #loop over the (x, y) coordinates and radii of the circles
        for (x, y, r) in circles:
            # Draw circles in the picture
            cv2.circle(img_undist, (x, y), r, (255, 255, 0), 3)   
    if DEBUG_MODE:
        print("The circles drawn should be coinciding with the markers in the cropped out picture")
        plt.figure(figsize=(200,200))
        plt.imshow(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    if PRODUCTION_MODE:
        x_max_height_column=int(input("Enter the cylinder number of maximum height"))
    if DEBUG_MODE:
        x_max_height_column = 38
    
    centers_columns = []
    for (x,y,r) in circles:
        centers_columns.append((x,y))
    centers_columns.sort()

    tube_numbers = np.array(range(no_of_cylinders))+1
    y_height = np.ones(no_of_cylinders)*img_bw.shape[0]
    
    y_pixel_coord = []
    x_pos = []

    for _,i in centers_columns:
        y_pixel_coord.append(i)
    temp_min_arg = np.argmin(np.array(y_pixel_coord))
    if DEBUG_MODE:
        print("The y pixels: ",y_pixel_coord)
        print(temp_min_arg)
    for i in range(len(y_pixel_coord)):
        y_height[i+x_max_height_column-temp_min_arg-1] = y_pixel_coord[i]
    
    print(np.argmin(y_height))
    
    ## Doing Simple Linear Regression
    if PRODUCTION_MODE:
        no_of_data_points = int(input("Enter the number of data points you want to use for conversion"))
        data_y = np.zeros(no_of_data_points)
        data_x_columns = np.zeros(no_of_data_points,dtype=np.uint16)
        for i in range(no_of_data_points):
            data_x_columns[i]  = int(input(f"Enter the column number for data point {(i+1)}"))
            data_y[i] = float(input(f"Enter the height in column {data_x_columns[i]} for data point {(i+1)}"))
    
    if DEBUG_MODE:
        no_of_data_points = 23
        # This needs to be filled with some values for debugging
        #Values for image 6.jpg
        data_y = [7.6,7.0,6.6,6.1,6.0,6.3,7.3,8.9,11.5,13.2,13.2,13.8,14.4,15,14.5,13,12.5,11.8,10.5,8.2,7,6.3,5.9]
        data_x_columns = [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]


    data_x = np.zeros(no_of_data_points)
    for i in range(no_of_data_points):
        data_x[i] = y_height[data_x_columns[i]-1]

    # from scipy.optimize import curve_fit
    # def objective(x,a,b,c,d):
    #     return c*x+ a*x**3 + b*x**2 + d
    # popt, _ = curve_fit(objective, data_x, data_y)
    data_x = data_x.reshape((-1,1))
    model = LinearRegression()
    model.fit(data_x, data_y)
    # a,b,c,d = popt

    if DEBUG_MODE:
        print("Linear Regression is a ML model")
        print(f"intercept: {model.intercept_}")
        print(f"slope: {model.coef_}")
        # print("Using Curve fitting")
        # print(f"a = {a}, b= {b}, c ={c}, d={d}")
    
    # In Case, u find some good combination which works
    # slope_to_use = 0
    # intercept_to_use = 0 
    slope_to_use = model.coef_
    intercept_to_use = model.intercept_
    y_height = y_height*slope_to_use + intercept_to_use

    # y_height = objective(y_height,a,b,c,d)

    x_pos = tube_numbers*each_cylinder_width_mm - each_cylinder_width_mm/2

    plt.figure(figsize=(16, 4))
    plt.scatter(x_pos,y_height)
    
    #### DENUG MODE ####
    if PRODUCTION_MODE:
        product_no = str(input('Product number: '))
        pressure = str(input('Pressure in bar: '))
        flowrate = str(input('Volume Flow in l/min: '))
        nozz_height = str(input('Horizontal Stand in mm: '))
    if DEBUG_MODE:
        product_no ='0'
        pressure ='0'
        flowrate ='0'
        nozz_height ='0'

    title_full = 'Liquid Distribution for '+ product_no + ' at '+ pressure +' bar // '+ flowrate+' l/min' 
    plt.xlim([0, np.max(x_pos)])
    plt.ylim([0, np.max(y_height)+1])
    plt.grid()
    # # plt.title('Liquid Distribution Measurement')
    plt.title(title_full)
    plt.xlabel('Position in mm')
    plt.ylabel('Level in mm')
    plt.show()

    if PRODUCTION_MODE:
        percent_deviation= int(input("Enter the percentage deviation threshold you want to see"))
        # percent_deviation = 15
    if DEBUG_MODE:
        percent_deviation= 15

    mean_distr = np.mean(y_height)
    deviation_greater = (y_height > mean_distr*(1 + percent_deviation/100)).sum()
    deviation_lesser = (y_height < mean_distr*(1 - percent_deviation/100)).sum()

    #-----------------------------------------------
    # Clean up the result and save it in xlsx 
    #-----------------------------------------------

    # Divide the x position into whole 15mm increments

    # #Create DataFrame from lists
    df = pd.DataFrame(list(zip(x_pos,y_height)), columns = ['xpos', 'level'])

    # # Sort DataFrame in ascending order by position
    df = df.sort_values(by=['xpos'])
    
    # # Scaling image for saving in excel
    img = img_undist.copy()
    height, width = img.shape[:2]
    max_height = 1000
    max_width = 1000

    # # only shrink if img is bigger than required
    if max_height < height or max_width < width:
    # get scaling factor
        scaling_factor = max_height / float(height)
    if max_width/float(width) < scaling_factor:
        scaling_factor = max_width / float(width)
    # resize image
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Clean up duplicate entries for x position
    df = df.drop_duplicates(['xpos'])
    
    if PRODUCTION_MODE:
        filename = str(input('File name result file without extension:'))
    if DEBUG_MODE:
        filename = "111"
    
    if filename == (""):
        filename = 'FLV_' + product_no + '_' + pressure +'_bar_' + flowrate + '_lpm_' + nozz_height + '_mm'
    else: 
        filename = filename
    filenamexlsx = './results/'+ filename + '.xlsx'
    filenameimg = './results/'+ filename + '.png'
    filenameimg_ext = './results/'+ filename + "_extracted" + '.png'
    filenametxt = './results/' + filename + '.txt'

    plt.figure(figsize=(16, 4))
    plt.scatter(x_pos,y_height)
    title_full = f"Liquid Distribution for {product_no} at {pressure} bar // { flowrate } l/min /n Deviation above {percent_deviation} = { deviation_greater} Deviation below {percent_deviation} ={deviation_lesser}"
    plt.xlim([0, np.max(x_pos)])
    plt.ylim([0, np.max(y_height)+1])
    plt.grid()
    # # plt.title('Liquid Distribution Measurement')
    plt.title(title_full)
    plt.xlabel('Position in mm')
    plt.ylabel('Level in mm')
    plt.axhline(y=mean_distr, color='r', linestyle='-')
    plt.axhline(y=mean_distr*(1 + percent_deviation/100),color = 'b', linestyle='dotted')
    plt.axhline(y=mean_distr*(1 - percent_deviation/100),color = 'b', linestyle='dotted')
    plt.savefig(filenameimg_ext)
    
    try:
        os.makedirs('./results')
    except OSError:
        pass
    # Data Frame und Bild  in excel schreiben
    writer = pd.ExcelWriter(filenamexlsx, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Measurement')
    plt.imsave(filenameimg, img)
    workbook = writer.book
    worksheet = writer.sheets['Measurement']
    worksheet.insert_image('E3', filenameimg)
    worksheet.insert_image('E10', filenameimg_ext)
    # writer.write('Measurement','E19',"LOLOL")
    writer.close()
    
    print('File saved in subfolder ./results. Program ended.')

    # Creating txt file
    with open(filenametxt, 'w') as f:
        for i in y_height:
            f.write(f"{i:.2f}\t")




    return img_undist


    


#### LEGACY CODE ####
# def find_height(img_raw,corresponding_height):
#     y_pix_sum = 0
#     y_pix_num = 0
#     for i in range(img_raw.shape[0]):
#         for j in range(img_raw.shape[1]):
#             if(img_raw[i][j]>128):
#                 y_pix_num+=1
#                 y_pix_sum+=i
#     if y_pix_num==0:
#         return 0
#     return y_pix_sum//y_pix_num


# def find_balls(img_raw, img_undist):
#     #++++++++++++++++++++++++++++++++
#     #+ Find balls on the measuring track +
#     #++++++++++++++++++++++++++++++++
#     #Perform Hough circle operation to find circles.
#     #minDist: Minimum distance to the next neighbor found
#     #minRadius, maxRadiusin pixels.
#     img_bw = img_raw.copy()
#     circles = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1.1, minDist=5, param1=5, param2=10, minRadius = 5, maxRadius = 15)

#     #Ensuring that something is found at all:
#     if circles is not None: 
#         #Convert to integer, otherwise it won't loop.
#         circles = np.round(circles[0, :]).astype("int")
#         #circles = np.round(circles[0, :]/25)*25.astype("int")
#         #loop over the (x, y) coordinates and radii of the circles
#         for (x, y, r) in circles:
#             # Draw circles in the picture
#             cv2.circle(img_raw, (x, y), r, (255, 255, 255), 3)
#             #Draw rectangles into the picture
#             #cv2.rectangle(img_color, (x -10, y -10), (x + 10, y + 10), (0, 0, 255), 2)

#     # # Plotten the FLV  
#     # #y coordinates: find max and subtract all values ​​from it for display "right side up"
#     # plt.figure(figsize=(16, 4))
#     # plt.scatter(circles[:,0], max(circles[:,1])-circles[:,1])
#     # yMax = max(circles [:,1])
#     # # product_no = str(input('Product number: '))
#     # product_no ='0'
#     # # pressure = str(input('Druck in bar: '))
#     # pressure ='0'
#     # # flowrate = str(input('Volume Flow in l/min: '))
#     # flowrate ='0' 
#     # # nozz_height = str(input('Horizontal Stand in mm: '))
#     # nozz_height ='0'
#     # title_full = 'Liquid Distribution for '+ product_no + ' at '+ pressure +' bar // '+ flowrate+' l/min' 
#     # plt.xlim([0, 3250])
#     # plt.ylim([0, 500])
#     # plt.grid()
#     # # plt.title('Liquid Distribution Measurement')
#     # plt.title(title_full)
#     # plt.xlabel('Position in mm')
#     # plt.ylabel('Level in mm')
#     # plt.show()
#     # #-----------------------------------------------
#     # # Clean up the result and save it in xlsx 
#     # #-----------------------------------------------

#     # # Divide the x position into whole 25mm increments
#     # circles_xpos = np.round(circles[:,0]/25,0)*25
#     # circles_ypos = max(circles[:,1])-circles[:,1]

#     # #Create DataFrame from lists
#     # df = pd.DataFrame(list(zip(circles_xpos,circles_ypos)), columns = ['xpos', 'level'])

#     # # Sort DataFrame in ascending order by position
#     # df = df.sort_values(by=['xpos'])
    
#     # # Scaling image for saving in excel
#     # img = img_undist.copy()
#     # height, width = img.shape[:2]
#     # max_height = 1000
#     # max_width = 1000

#     # # only shrink if img is bigger than required
#     # if max_height < height or max_width < width:
#     # # get scaling factor
#     #     scaling_factor = max_height / float(height)
#     # if max_width/float(width) < scaling_factor:
#     #     scaling_factor = max_width / float(width)
#     # # resize image
#     # img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

#     # # Clean up duplicate entries for x position
#     # df = df.drop_duplicates(['xpos'])
    
#     # # filename = str(input('File name result file without extension:'))
#     # filename = "111"
#     # if filename == (""):
#     #     filename = 'FLV_' + product_no + '_' + pressure +'_bar_' + flowrate + '_lpm_' + nozz_height + '_mm'
#     # else: 
#     #     filename = filename #+ '.xlsx'
    
#     # filenamexlsx = './results/'+ filename + '.xlsx'
#     # filenameimg = './results/'+ filename + '.png'
#     # try:
#     #     os.makedirs('./results')
#     # except OSError:
#     #     pass
#     # # Data Frame und Bild  in excel schreiben
#     # writer = pd.ExcelWriter(filenamexlsx, engine='xlsxwriter')
#     # df.to_excel(writer, sheet_name='Measurement')
#     # plt.imsave(filenameimg, img)
#     # workbook = writer.book
#     # worksheet = writer.sheets['Measurement']
#     # worksheet.insert_image('E3', filenameimg)
#     # writer.close()
    
#     # print('File saved in subfolder ./results. Program ended.')




# # We use the undistorted cropped image to see the cylinders and their centers
#     _,lines = find_columns(img_undist)
#     # 20 shud lie in the y coordinates of lines
#     # print(lines)

#     possible_lines = []
#     for l in lines:
#         if (l[0][1]<20 and l[0][3]>20) or (l[0][1]>20 and l[0][3]<20) and l[0][2] == l[0][0]:
#             possible_lines.append(l[0][2])
#     possible_lines.sort()
#     print(possible_lines)
#     # cv2.circle(img_undist,(possible_lines[3] ,50),color=(255),radius=5,thickness=2)
#     bins = []
#     temp = [possible_lines[0]]
#     for i in possible_lines[1:]:
#         if (i-temp[-1])>each_cylinder_width_pix//2:
#             bins.append(temp)
#             temp = [i]
#         else:
#             temp.append(i)
#     bins.append(temp)
#     x_coor = []
#     for x in bins:
#         x_coor.append(np.int32(np.round(np.mean(x))))
#     print(x_coor)
#     y_height = []
#     for i in x_coor:
#         temp_sum =0
#         temp_num =0
#         for l in range(i-10,i+10):
#             for j in range(20,img_bw.shape[0]*24//25):
#                 if( img_bw[j][l]>10):
#                     temp_sum+=j
#                     temp_num+=1
#         if temp_num==0:
#             y_height.append(img_bw.shape[0])
#         else:
#             y_height.append(img_bw.shape[0]-temp_sum/temp_num)