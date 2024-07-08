import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import aruco
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
import yaml
import liqdist_archit as ld

class ArUcoDetector:
    def __init__(self, video_path, cooldown_time, output_dir="output_frames"):
        self.video_path = video_path
        self.vid = cv2.VideoCapture(video_path)
        self.cooldown_time = cooldown_time
        self.output_dir = output_dir
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        self.max_area = 0
        self.max_frame = None
        self.frame_id = 1 
        self.frame_counter = 0
        self.cooldown_frames = int(30 * cooldown_time)
        self.timestamps = []
        self.areas = []
        self.previous_area = 0
        self.previous_previous_area = 0

    def detect_aruco_closest_frame(self):
        while self.vid.isOpened():
            ret, frame = self.vid.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None and len(ids) == 4:
                current_area = self.compute_marker_area(corners)
                timestamp = self.vid.get(cv2.CAP_PROP_POS_MSEC)
                self.timestamps.append(timestamp / 1000)
                self.areas.append(current_area)

                if self.previous_area > self.previous_previous_area and self.previous_area > current_area and self.frame_counter >= self.cooldown_frames and self.previous_area > 260000:
                    self.max_frame = frame.copy()
                    print(f"Local max area found at {timestamp / 1000:.1f}s with area {self.previous_area:.1f} pixels")
                    self.save_frame(self.max_frame, self.frame_id)
                    self.frame_id += 1  
                    self.frame_counter = 0

                self.previous_previous_area = self.previous_area
                self.previous_area = current_area

            self.frame_counter += 1

        self.vid.release()
        cv2.destroyAllWindows()
        self.plot_areas_vs_time(filename)

    def compute_marker_area(self, corners):
        centers = [np.mean(corner, axis=1)[0] for corner in corners]
        centroid = np.mean(centers, axis=0)
        angles = [np.arctan2(center[1] - centroid[1], center[0] - centroid[0]) for center in centers]
        centers = [center for _, center in sorted(zip(angles, centers))]
        centers.append(centers[0])
        return 0.5 * abs(sum(x1 * y2 - x2 * y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))

    def save_frame(self, frame, frame_id):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filename = os.path.splitext(os.path.basename(self.video_path))[0]
        cv2.imwrite(f'{self.output_dir}/max_area_frame_{filename}_{frame_id}.png', frame)
    
    def plot_areas_vs_time(self, filename):
        # Convert lists to a pandas DataFrame
        data = pd.DataFrame({
            'Timestamp': self.timestamps,
            'Area': self.areas
        })
        
        # Export to CSV
        data.to_csv(f'areas_vs_time_{filename}.csv', index=False)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.areas, marker='o', linestyle='-', color='b')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Area (pixels)')
        plt.title('Area of ArUco Markers vs. Time')
        plt.grid(True)
        plt.savefig(f'areas_vs_time_{filename}.png')
        plt.show()

class CSVCombiner:
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def combine_csv_files(self):
        # Initialize an empty list to store the DataFrames
        dfs = []

        # Process each file in the directory and its subdirectories
        for root, dirs, files in os.walk(self.dir_name):
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

class Visualizer:
    @staticmethod
    def visualize_nozzle_distribution_3d(df, interpolation_method='nearest', colorscale='Viridis'):
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

    @staticmethod
    def csv_to_2D_Heatmap(df, colorscale='Jet'):
        scaler = MinMaxScaler(feature_range=(6, 14))
        Z = scaler.fit_transform(df['z'].values.reshape(-1, 1)).flatten()

        x = np.linspace(df['x'].min(), df['x'].max(), 500)
        y = np.linspace(df['y'].min(), df['y'].max(), 500)
        X, Y = np.meshgrid(x, y)
        Z = griddata((df['x'], df['y']), Z, (X, Y), method='cubic')

        contour = go.Contour(x=x, y=y, z=Z, colorscale=colorscale)
        fig = go.Figure(data=[contour])
        fig.update_layout(
            title='Heatmap of Z values',
            xaxis_title='X',
            yaxis_title='Y'
        )
        return fig

class ImageProcessor:
    def __init__(self, DEBUG_MODE=True):
        self.DEBUG_MODE = DEBUG_MODE
        self.PRODUCTION_MODE = not DEBUG_MODE   

    def process_images(self):
        
        if self.DEBUG_MODE:
            
            print("You are in debugging mode.")
            print("Multiple input streams are not supported")
            print("There will be lots of intermediate steps being printed out")

        ld.DEBUG_MODE = self.DEBUG_MODE
        ld.PRODUCTION_MODE = self.PRODUCTION_MODE

        image_files = os.listdir("output_frames")

        for image_file in image_files:
            filename = os.path.splitext(image_file)[0]
            output_dir = os.path.join("intermediate_outputs", filename)
            os.makedirs(output_dir, exist_ok=True)

            capture = cv2.imread(os.path.join("output_frames", image_file))
            camera_matrix, distortion_coefficients = ld.read_cam_calibration()

            if self.DEBUG_MODE:
                print("Image captured")
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "image_captured.png"))

            img_intrinsic = ld.intrinsic(capture, camera_matrix, distortion_coefficients)
            img_intrinsic = capture

            if self.DEBUG_MODE:
                print("Image after Undistortion")
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(img_intrinsic, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "image_after_undistortion.png"))

            arucoFound = ld.detect_arucos(capture, camera_matrix, distortion_coefficients)
            if self.DEBUG_MODE:
                if arucoFound is not None:
                    print("No of Aruco found: ", len(arucoFound))
                print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
                print("The detected arucos are: ", arucoFound)

            img_cr = ld.crop_image(img_intrinsic, arucoFound)
            if self.DEBUG_MODE:
                print("Cropped Images")
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(img_cr, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "cropped_image.png"))

            img_raw = ld.morphologic(img_cr)
            if self.DEBUG_MODE:
                print("Image Morphed")
                plt.figure(figsize=(10, 10))
                plt.imshow(img_raw)
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "image_morphed.png"))

            balls_found = ld.find_balls(img_raw, img_cr, output_dir, filename, count=filename[-2])

            cv2.imwrite(os.path.join(output_dir, "balls_found.png"), balls_found)

if __name__ == "__main__":
    filename = "../Vids/sop6.mp4"
    detector = ArUcoDetector(f'Vids/{filename}', cooldown_time=20)
    detector.detect_aruco_closest_frame()
    
    image_processor = ImageProcessor(DEBUG_MODE=True)
    image_processor.process_images()

    combiner = CSVCombiner(dir_name='intermediate_outputs')
    combined_df = combiner.combine_csv_files()

    visualizer = Visualizer()
    fig_3d = visualizer.visualize_nozzle_distribution_3d(combined_df)
    fig_3d.show()

    # fig_heatmap = visualizer.csv_to_2D_Heatmap(combined_df)
    # fig_heatmap.show()

    
