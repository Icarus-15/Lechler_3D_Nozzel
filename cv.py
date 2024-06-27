
# Note TO self never use OPENCV(PYTHON) in jupyter notebook it will crash the kernel on a mac 
# Use a python script instead

# Import the necessary libraries
import cv2 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata, UnivariateSpline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd 
import cv2.aruco as aruco 
import liqdist_archit as liqdist
import liqdist_archit as ld
import time
import yaml 
from liqdist_archit import detect_arucos
import numpy as np
from cv2 import aruco

# Example usage
filename = "sop4.mp4"
vid = cv2.VideoCapture(f'Vids/{filename}')

def detect_aruco_closest_frame(vid, output_dir="output_frames", cooldown_time=5):
    max_area = 0
    max_frame = None
    frame_counter = 0
    frame_id = 1 
    cooldown_frames = int(30 * cooldown_time)  # Assuming 30 FPS

    # Load ArUco dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    aruco_params = aruco.DetectorParameters()

    previous_area = 0
    previous_previous_area = 0

    timestamps = []
    areas = []

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) == 4:
            current_area = compute_marker_area(corners)

            # Store timestamp and area for plotting
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp / 1000)  # Convert to seconds
            areas.append(current_area)

            # Check for local maximum
            if previous_area > previous_previous_area and previous_area > current_area and frame_counter >= cooldown_frames and previous_area > 260000:
                max_frame = frame.copy()
                print(f"Local max area found at {timestamp / 1000:.1f}s with area {previous_area:.1f} pixels")
                             
                # Save the frame
                save_frame(max_frame, output_dir, filename, previous_area, frame_id)
                frame_id += 1  # Increment the frame ID for the next save
                frame_counter = 0  # Reset cooldown

            # Update areas
            previous_previous_area = previous_area
            previous_area = current_area

        frame_counter += 1  # Increment frame counter

    vid.release()
    cv2.destroyAllWindows()

    # Plot the areas vs timestamps
    plot_areas_vs_time(timestamps, areas, filename)

def compute_marker_area(corners):
    centers = [np.mean(corner, axis=1)[0] for corner in corners]
    centroid = np.mean(centers, axis=0)
    angles = [np.arctan2(center[1] - centroid[1], center[0] - centroid[0]) for center in centers]
    centers = [center for _, center in sorted(zip(angles, centers))]
    centers.append(centers[0])  # Close the loop
    return 0.5 * abs(sum(x1 * y2 - x2 * y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))

def save_frame(frame, output_dir, video_name, area, frame_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.splitext(video_name)[0]
    cv2.imwrite(f'{output_dir}/max_area_frame_{filename}_{frame_id}.png', frame)
    

def plot_areas_vs_time(timestamps, areas, filename):
    # Convert lists to a pandas DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Area': areas
    })
    
    # Export to CSV
    data.to_csv(f'areas_vs_time_{filename}.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, areas, marker='o', linestyle='-', color='b')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Area (pixels)')
    plt.title('Area of ArUco Markers vs. Time')
    plt.grid(True)
    plt.savefig(f'areas_vs_time_{filename}.png')
    plt.show()

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
    
def detect_aruco(vid, dictionary, parameters):
    max_area = 0
    max_frame = None
    offset_y = 10  # Define the offset for y
    count = 0 # Initialize the count to 0
    cam_mat , dist_coeff = liqdist.read_cam_calibration()
    frame_counter = 0  # Initialize a frame counter
    cooldown_frames = 30*8 # Set the number of frames to skip after capturing a frame

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        skewness = skew(gray.ravel())

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
        #frame = liqdist.intrinsic(frame, cam_mat, dist_coeff) 

        # If 4 markers are detected, compute the area
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
            # Compute the area using the Shoelace formula
            area = 0.5 * abs(sum(x1*y2 - x2*y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))
            # If the area is the maximum so far, save the frame and output the timestamp
            
            if area > max_area  and frame_counter >= cooldown_frames :
                max_area = area
                max_frame = frame.copy()
                timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)  # Get the timestamp of the current frame
                print(f"New max area found at {round((timestamp/1000)/60,1)} min {round((timestamp/1000) % 60,1)} s and the area is {area} pixels") 

                # Specify the directory name
                dir_name = "output_frames"

                # Check if the directory exists
                if not os.path.exists(dir_name):
                    # If the directory does not exist, create it
                    os.makedirs(dir_name)

                # Save the frame with the maximum area after the video has been processed
                if max_frame is not None:
                    cv2.imwrite(f'output_frames/max_area_frame_{os.path.splitext(filename)[0]}_{count}.png', max_frame)
                    max_frame = None # Reset the max_frame variable
                    count += 1  # Increment the count
                    frame_counter = 0  # Reset the frame counter

        frame_counter += 1  # Increment the frame counter for each frame

    vid.release()
    cv2.destroyAllWindows()




# def combine_csv_files(start, step, dir_name):
#     # Initialize an empty list to store the DataFrames
#     dfs = []

#     # Initialize the y value
#     y = start

#     # Process each file in the directory and its subdirectories
#     for root, dirs, files in os.walk(dir_name):
#         for file in sorted(files):
#             # Check if the file is a CSV file and not the combined output file
#             if file.endswith('.csv') and file != 'combined_green_balls.csv':
#                 # Load the CSV file
#                 csv_path = os.path.join(root, file)
#                 df = pd.read_csv(csv_path)

#                 # Check for existing 'y' column and print a warning if found
#                 if 'y' in df.columns:
#                    # print(f"Warning: 'y' column already exists in {csv_path}. It will be overwritten.")
#                     pass
#                 # Add the y column
#                 df['y'] = y

#                 # Append the DataFrame to the list
#                 dfs.append(df)

#         # Increment the y value for each new file
#         y += step

#                 # Print debug information
#                 #print(f"Processed file: {csv_path}, y value assigned: {y - step}")

#     # Concatenate the DataFrames
#     combined_df = pd.concat(dfs, ignore_index=True)

#     # Save the combined DataFrame to a CSV file
#     combined_csv_path = os.path.join(dir_name, 'combined_green_balls.csv')
#     combined_df.to_csv(combined_csv_path, index=False)
    
#     # # Debug print for count of y = 1
#     # count_y_1 = len(combined_df[combined_df['y'] == 1])
#     # print(f"Number of rows with y = 1: {count_y_1}")

#     return combined_df

def combine_csv_files(dir_name):
    # Initialize an empty list to store the DataFrames
    dfs = []

    # Process each file in the directory and its subdirectories
    for root, dirs, files in os.walk(dir_name):
        for file in sorted(files):
            # Check if the file is a CSV file and not the combined output file
            if file.endswith('.csv') and file != 'combined_green_balls.csv':
                # Load the CSV file
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)

                # Append the DataFrame to the list
                dfs.append(df)

    # Concatenate the DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

def visualize_3D_distribution(df, colorscale='Jet'):

    # Extract the x, y, and z values
    X = df['x'].values
    Y = df['y'].values
    Z = df['z'].values
    
    for i in Y:
        max_z = df[df['y'] == i]['z'].max()
        print(f"Maximum Z value for y = {i}: {max_z}")

    # Create a 3D line (mesh) plot
    fig = go.Figure(data=[go.Mesh3d(x=X, y=Y, z=Z, intensity=Z, colorscale=colorscale)])
    
    fig.update_layout(
        title = "Lechler Nozzel Distribution",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        ),
       # width=700,
        #margin=dict(r=20, l=10, b=10, t=10)
    )
    
    return fig
    
def csv_to_2D_Heatmap(df, colorscale='Jet'):
    
   # Scale Z values to be between 6 and 14
    scaler = MinMaxScaler(feature_range=(6, 14))
    Z = scaler.fit_transform(df['z'].values.reshape(-1, 1)).flatten()

    # Create a grid of points in the x, y plane
    x = np.linspace(df['x'].min(), df['x'].max(), 500)
    y = np.linspace(df['y'].min(), df['y'].max(), 500)
    X, Y = np.meshgrid(x, y)

    # Interpolate Z values on this grid using cubic interpolation
    Z = griddata((df['x'], df['y']), Z, (X, Y), method='cubic')
    # Create a heatmap
    contour = go.Contour(
        x=x,
        y=y,
        z=Z,
        colorscale=colorscale
    )

    # Create a figure and add the heatmap
    fig = go.Figure(data=[contour])

    # Set layout properties
    fig.update_layout(
        title='Heatmap of Z values',
        xaxis_title='X',
        yaxis_title='Y',
    )

    return fig

def process_images(PRODUCTION_MODE=False, DEBUG_MODE=True):
    import cv2
    import numpy as np
    import yaml 
    import liqdist_archit as ld
    from liqdist_archit import detect_arucos
    import os

    ld.DEBUG_MODE      = DEBUG_MODE
    ld.PRODUCTION_MODE = PRODUCTION_MODE

    if DEBUG_MODE:
        import matplotlib.pyplot as plt
        print("You are in debugging mode.")
        print("Multiple input streams are not supported")
        print("There will be lots of intermediate steps being printed out")

    image_files = os.listdir("output_frames")

    for image_file in image_files:
        filename = os.path.splitext(image_file)[0] 
        output_dir = os.path.join("intermediate_outputs", filename)
        os.makedirs(output_dir, exist_ok=True)

        capture = cv2.imread(os.path.join("output_frames", image_file))

        camera_matrix, distortion_coefficients = ld.read_cam_calibration()

        if DEBUG_MODE:
            print("Image captured")
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "image_captured.png"))

        img_intrinsic = ld.intrinsic(capture,camera_matrix,distortion_coefficients)

        img_intrinsic = capture
        if DEBUG_MODE:
            print("Image after Undistortion")
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(img_intrinsic, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "image_after_undistortion.png"))

        arucoFound = ld.detect_arucos(capture,camera_matrix,distortion_coefficients)
        if DEBUG_MODE:
            if arucoFound is not None:
                print("No of Aruco found: ",len(arucoFound))
            print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
            print("The detected arucos are: ",arucoFound)

        img_cr = ld.crop_image(img_intrinsic,arucoFound)
        if DEBUG_MODE:
            print("Cropped Images")
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(img_cr, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "cropped_image.png"))

        img_raw = ld.morphologic(img_cr)
        if DEBUG_MODE:
            print("Image Morphed")
            plt.figure(figsize=(10,10))
            plt.imshow(img_raw)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "image_morphed.png"))

        balls_found = ld.find_balls(img_raw, img_cr, output_dir,filename)

        cv2.imwrite(os.path.join(output_dir, "balls_found.png"), balls_found)

def radial_gaussian(xy, amplitude, x0, y0, sigma):
    x, y = xy
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    return amplitude * np.exp(-r**2 / (2 * sigma**2))

def visualize_3D_cone(df, colorscale='Jet', noise_level=0.05):

    # Extract the x, y, and z values
    X = df['x'].values
    Y = df['y'].values
    Z = df['z'].values

    # Assuming the apex of the cone is at the maximum Z value
    apex = [np.mean(X), np.mean(Y), Z.max()]

    # Generate cone points (simplified approach)
    cone_radius = max(np.ptp(X), np.ptp(Y)) / 2  # Cone base radius
    cone_height = Z.max() - Z.min()  # Cone height
    t = np.linspace(0, 2 * np.pi, 30)
    h = np.linspace(0, cone_height, 20)
    t, h = np.meshgrid(t, h)
    X_cone = apex[0] + (cone_radius * h/cone_height) * np.cos(t)
    Y_cone = apex[1] + (cone_radius * h/cone_height) * np.sin(t)
    Z_cone = apex[2] - h

    # Add noise
    X_cone += np.random.normal(0, noise_level, X_cone.shape)
    Y_cone += np.random.normal(0, noise_level, Y_cone.shape)
    Z_cone += np.random.normal(0, noise_level, Z_cone.shape)

    # Project heatmap onto cone surface
    # This is a simplified approach; a more accurate method would involve calculating
    # the distance of each cone point to the nearest data point and adjusting intensity accordingly
    intensity = np.sqrt((X_cone - apex[0])**2 + (Y_cone - apex[1])**2 + (Z_cone - apex[2])**2)
    intensity = intensity.flatten()
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())  # Normalize

    # Create a 3D plot with the cone and projected heatmap
    fig = go.Figure(data=[
        go.Mesh3d(x=X_cone.flatten(), y=Y_cone.flatten(), z=Z_cone.flatten(), intensity=intensity, colorscale=colorscale, opacity=0.5, name='Cone with Heatmap')
    ])
    # Create a 3D line (mesh) plot
    fig = go.Figure(data=[go.Mesh3d(x=X, y=Y, z=Z, intensity=Z, colorscale=colorscale)])
    fig.update_layout(
        title="Lechler Nozzle Distribution with Projected Heatmap on Cone",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        )
    )

    return fig

def visualize_nozzle_distribution_3d(df, interpolation_method='cubic', colorscale='Viridis'):
    """
    Visualize the water level distribution of a full cone nozzle as a 3D surface plot.
    
    :param df: DataFrame containing 'x', 'y', 'z' columns where z represents the water level
    :param interpolation_method: Method used for interpolation ('linear', 'nearest', or 'cubic')
    :param colorscale: Colorscale for the plot
    :return: Plotly figure object
    """
    # Create a grid for interpolation
    xi = np.linspace(df['x'].min(), df['x'].max(), 100)
    yi = np.linspace(df['y'].min(), df['y'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the water level for the 3D surface
    water_level_surface = griddata(
        (df['x'], df['y']), 
        df['z'],
        (xi, yi),
        method=interpolation_method
    )

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=xi, y=yi, z=water_level_surface,
        colorscale=colorscale,
        colorbar=dict(title='Water Level')
    )])

    # Update layout
    fig.update_layout(
        title="Full Cone Nozzle Water Level Distribution",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Water Level',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=800,
        height=800,
    )

    return fig
#detect_aruco_closest_frame(vid, cooldown_time=20)


# For the 2D heatmap 
#heatmap = csv_to_2D_Heatmap(combine_csv_files(1, 1, dir_name='intermediate_outputs'))
#heatmap.show()





#### Set System Mode
PRODUCTION_MODE = False
DEBUG_MODE = True


ld.DEBUG_MODE      = DEBUG_MODE
ld.PRODUCTION_MODE = PRODUCTION_MODE

if DEBUG_MODE:
    import matplotlib.pyplot as plt
    print("You are in debugging mode.")
    print("Multiple input streams are not supported")
    print("There will be lots of intermediate steps being printed out")

# Get a list of all the images in the output frames folder
image_files = os.listdir("output_frames")
counter = 1 

for image_file in image_files:
    # Create a new directory for the intermediate outputs of this image
    filename = os.path.splitext(image_file)[0] 
    # Create a new directory for the intermediate outputs of this image
    output_dir = os.path.join("intermediate_outputs", filename)
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    capture = cv2.imread(os.path.join("output_frames", image_file))

    #capture, selector = ld.select_ipcamera()
    camera_matrix, distortion_coefficients = ld.read_cam_calibration()

    if DEBUG_MODE:
        print("Image captured")
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "image_captured.png"))

    img_intrinsic = ld.intrinsic(capture,camera_matrix,distortion_coefficients)

    img_intrinsic = capture
    if DEBUG_MODE:
        print("Image after Undistortion")
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(img_intrinsic, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "image_after_undistortion.png"))

    arucoFound = ld.detect_arucos(capture,camera_matrix,distortion_coefficients)
    if DEBUG_MODE:
        if arucoFound is not None:
            print("No of Aruco found: ",len(arucoFound))
        print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
        print("The detected arucos are: ",arucoFound)

    img_cr = ld.crop_image(img_intrinsic,arucoFound)
    if DEBUG_MODE:
        print("Cropped Images")
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(img_cr, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "cropped_image.png"))

    img_raw = ld.morphologic(img_cr)
    if DEBUG_MODE:
        print("Image Morphed")
        plt.figure(figsize=(10,10))
        plt.imshow(img_raw)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "image_morphed.png"))
    
    balls_found = ld.find_balls(img_raw, img_cr, output_dir,filename,count = filename[-1])


    # Save the final output
    cv2.imwrite(os.path.join(output_dir, "balls_found.png"), balls_found)
    
    
# For the 3D distribution
#plot_3d = visualize_nozzle_distribution_3d(combine_csv_files(dir_name='intermediate_outputs'))
plot_3d = visualize_3D_distribution(combine_csv_files(dir_name='intermediate_outputs')) 
plot_3d.show()