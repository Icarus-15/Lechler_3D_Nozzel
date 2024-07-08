import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import aruco
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import yaml
import liqdist_archit as ld

class Config:
    def __init__(self, debug_mode=False):
        self.DEBUG_MODE = debug_mode
        self.PRODUCTION_MODE = not debug_mode

class ArUcoDetector:
    def __init__(self, video_path, cooldown_time, config, output_dir="output_frames"):
        self.video_path = video_path
        self.vid = cv2.VideoCapture(video_path)
        self.cooldown_time = cooldown_time
        self.output_dir = output_dir
        self.config = config
        self.setup_aruco()
        self.initialize_variables()

    def setup_aruco(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()

    def initialize_variables(self):
        self.max_area = 0
        self.max_frame = None
        self.frame_id = 1
        self.frame_counter = 0
        self.cooldown_frames = int(30 * self.cooldown_time)
        self.timestamps = []
        self.areas = []
        self.previous_area = 0
        self.previous_previous_area = 0

    def detect_aruco_closest_frame(self):
        while self.vid.isOpened():
            ret, frame = self.vid.read()
            if not ret:
                break

            self.process_frame(frame)

        self.cleanup()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) == 4:
            self.handle_valid_frame(frame, corners)

        self.frame_counter += 1

    def handle_valid_frame(self, frame, corners):
        current_area = self.compute_marker_area(corners)
        timestamp = self.vid.get(cv2.CAP_PROP_POS_MSEC)
        self.timestamps.append(timestamp / 1000)
        self.areas.append(current_area)

        if self.is_local_maximum(current_area):
            self.save_max_frame(frame, timestamp)

        self.update_previous_areas(current_area)

    def is_local_maximum(self, current_area):
        return (self.previous_area > self.previous_previous_area and
                self.previous_area > current_area and
                self.frame_counter >= self.cooldown_frames and
                self.previous_area > 260000)

    def save_max_frame(self, frame, timestamp):
        self.max_frame = frame.copy()
        print(f"Local max area found at {timestamp / 1000:.1f}s with area {self.previous_area:.1f} pixels")
        if self.config.DEBUG_MODE:
            self.save_frame(self.max_frame, self.frame_id)
        self.frame_id += 1
        self.frame_counter = 0

    def update_previous_areas(self, current_area):
        self.previous_previous_area = self.previous_area
        self.previous_area = current_area

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

    def cleanup(self):
        self.vid.release()
        cv2.destroyAllWindows()
        if self.config.DEBUG_MODE:
            self.plot_areas_vs_time()

    def plot_areas_vs_time(self):
        data = pd.DataFrame({
            'Timestamp': self.timestamps,
            'Area': self.areas
        })
        
        if self.config.DEBUG_MODE:
            filename = os.path.splitext(os.path.basename(self.video_path))[0]
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
        dfs = []
        for root, dirs, files in os.walk(self.dir_name):
            for file in sorted(files):
                if file.endswith('.csv') and file != 'combined_green_balls.csv':
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

class Visualizer:
    @staticmethod
    def visualize_nozzle_distribution_3d(df, interpolation_method='cubic', colorscale='Viridis'):
        df_filtered = Visualizer.filter_and_shift_data(df)
        xi, yi = Visualizer.create_grid(df_filtered)
        water_level_surface = Visualizer.interpolate_water_level(df_filtered, xi, yi, interpolation_method)
        fig = Visualizer.create_3d_surface_plot(xi, yi, water_level_surface, colorscale)
        return fig

    @staticmethod
    def filter_and_shift_data(df):
        # Step 1: Group by y values and find the minimum z for each group
        min_z_per_y = df.groupby('y')['z'].min()

        # Step 2: Filter out z values below the threshold for each y
        df_filtered = pd.DataFrame()
        for y_value, group in df.groupby('y'):
            threshold = min_z_per_y[y_value]
            filtered_group = group[group['z'] > threshold]
            df_filtered = pd.concat([df_filtered, filtered_group])

        # Step 3: Subtract the minimum z for each y value
        for y_value, group in df_filtered.groupby('y'):
            df_filtered.loc[group.index, 'z'] -= min_z_per_y[y_value]

        return df_filtered


    @staticmethod
    def create_grid(df):
        xi = np.linspace(df['x'].min(), df['x'].max(), 100)
        yi = np.linspace(df['y'].min(), df['y'].max(), 100)
        return np.meshgrid(xi, yi)

    @staticmethod
    def interpolate_water_level(df, xi, yi, interpolation_method):
        return griddata(
            (df['x'], df['y']), 
            df['z'],
            (xi, yi),
            method=interpolation_method
        )

    @staticmethod
    def create_3d_surface_plot(xi, yi, water_level_surface, colorscale):
        fig = go.Figure(data=[go.Surface(
            x=xi, y=yi, z=water_level_surface,
            colorscale=colorscale,
            colorbar=dict(title='Water Level')
        )])
        fig.update_layout(
            title="Full Cone Nozzle Water Level Distribution Above Threshold",
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
        df_filtered = Visualizer.filter_and_shift_data(df)
        Z = Visualizer.scale_z_values(df_filtered)
        x, y, Z = Visualizer.prepare_heatmap_data(df_filtered, Z)
        fig = Visualizer.create_heatmap(x, y, Z, colorscale)
        return fig

    @staticmethod
    def scale_z_values(df):
        scaler = MinMaxScaler(feature_range=(6, 14))
        return scaler.fit_transform(df['z'].values.reshape(-1, 1)).flatten()

    @staticmethod
    def prepare_heatmap_data(df, Z):
        x = np.linspace(df['x'].min(), df['x'].max(), 500)
        y = np.linspace(df['y'].min(), df['y'].max(), 500)
        X, Y = np.meshgrid(x, y)
        Z = griddata((df['x'], df['y']), Z, (X, Y), method='cubic')
        return x, y, Z

    @staticmethod
    def create_heatmap(x, y, Z, colorscale):
        contour = go.Contour(x=x, y=y, z=Z, colorscale=colorscale)
        fig = go.Figure(data=[contour])
        fig.update_layout(
            title='Heatmap of Z values Above Threshold',
            xaxis_title='X',
            yaxis_title='Y'
        )
        return fig


class ImageProcessor:
    def __init__(self, config):
        self.config = config

    def process_images(self):
        if self.config.DEBUG_MODE:
            self.print_debug_info()

        ld.DEBUG_MODE = self.config.DEBUG_MODE
        ld.PRODUCTION_MODE = self.config.PRODUCTION_MODE

        image_files = os.listdir("output_frames")

        for image_file in image_files:
            self.process_single_image(image_file)

    def print_debug_info(self):
        print("You are in debugging mode.")
        print("Multiple input streams are not supported")
        print("There will be lots of intermediate steps being printed out")

    def process_single_image(self, image_file):
        filename = os.path.splitext(image_file)[0]
        output_dir = os.path.join("intermediate_outputs", filename)
        if self.config.DEBUG_MODE:
            os.makedirs(output_dir, exist_ok=True)

        capture = cv2.imread(os.path.join("output_frames", image_file))
        camera_matrix, distortion_coefficients = ld.read_cam_calibration()

        if self.config.DEBUG_MODE:
            self.save_debug_image(capture, output_dir, "image_captured.png")

        img_intrinsic = self.perform_intrinsic_calibration(capture, camera_matrix, distortion_coefficients)

        if self.config.DEBUG_MODE:
            self.save_debug_image(img_intrinsic, output_dir, "image_after_undistortion.png")

        arucoFound = ld.detect_arucos(capture, camera_matrix, distortion_coefficients)
        if self.config.DEBUG_MODE:
            self.print_aruco_info(arucoFound)

        img_cr = ld.crop_image(img_intrinsic, arucoFound)
        if self.config.DEBUG_MODE:
            self.save_debug_image(img_cr, output_dir, "cropped_image.png")

        img_raw = ld.morphologic(img_cr)
        if self.config.DEBUG_MODE:
            self.save_debug_image(img_raw, output_dir, "image_morphed.png", is_gray=True)
        
        count_id = filename.split("_")[-1].split(".")[0]
        balls_found = ld.find_balls(img_raw, img_cr, output_dir, filename, count=count_id)

        if self.config.DEBUG_MODE:
            cv2.imwrite(os.path.join(output_dir, "balls_found.png"), balls_found)

    def save_debug_image(self, img, output_dir, filename, is_gray=False):
        plt.figure(figsize=(10, 10))
        if not is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, cmap='gray' if is_gray else None)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, filename))

    def perform_intrinsic_calibration(self, capture, camera_matrix, distortion_coefficients):
        return ld.intrinsic(capture, camera_matrix, distortion_coefficients)

    def print_aruco_info(self, arucoFound):
        if arucoFound is not None:
            print("No of Aruco found: ", len(arucoFound))
        print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
        print("The detected arucos are: ", arucoFound)

def main(filename, debug_mode=True):
    config = Config(debug_mode=debug_mode)
    
    detector = ArUcoDetector(f'{filename}', cooldown_time=20, config=config)
    detector.detect_aruco_closest_frame()
    
    image_processor = ImageProcessor(config)
    image_processor.process_images()

    combiner = CSVCombiner(dir_name='intermediate_outputs')
    combined_df = combiner.combine_csv_files()

    visualizer = Visualizer()
    fig_3d = visualizer.visualize_nozzle_distribution_3d(combined_df)
    fig_3d.show()

    # Uncomment the following lines if you want to generate and show the 2D heatmap
    # fig_heatmap = visualizer.csv_to_2D_Heatmap(combined_df)
    # fig_heatmap.show()

if __name__ == "__main__":
    main(filename="../Vids/sop5_1.mp4", debug_mode=True)