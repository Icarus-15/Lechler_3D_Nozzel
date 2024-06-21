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

class ArUcoDetector:
    def __init__(self, video_path, cooldown_time=5, output_dir="output_frames"):
        self.video_path = video_path
        self.vid = cv2.VideoCapture(video_path)
        self.cooldown_time = cooldown_time
        self.output_dir = output_dir
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        self.aruco_params = aruco.DetectorParameters()
        self.max_area = 0
        self.max_frame = None
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

                if self.previous_area > self.previous_previous_area and self.previous_area > current_area and self.frame_counter >= self.cooldown_frames:
                    self.max_frame = frame.copy()
                    print(f"Local max area found at {timestamp / 1000:.1f}s with area {self.previous_area:.1f} pixels")
                    self.save_frame(self.max_frame, self.previous_area)
                    self.frame_counter = 0

                self.previous_previous_area = self.previous_area
                self.previous_area = current_area

            self.frame_counter += 1

        self.vid.release()
        cv2.destroyAllWindows()
        self.plot_areas_vs_time()

    def compute_marker_area(self, corners):
        centers = [np.mean(corner, axis=1)[0] for corner in corners]
        centroid = np.mean(centers, axis=0)
        angles = [np.arctan2(center[1] - centroid[1], center[0] - centroid[0]) for center in centers]
        centers = [center for _, center in sorted(zip(angles, centers))]
        centers.append(centers[0])
        return 0.5 * abs(sum(x1 * y2 - x2 * y1 for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])))

    def save_frame(self, frame, area):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filename = os.path.splitext(os.path.basename(self.video_path))[0]
        cv2.imwrite(f'{self.output_dir}/max_area_frame_{filename}_{area:.1f}.png', frame)

    def plot_areas_vs_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.areas, marker='o', linestyle='-', color='b')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Area (pixels)')
        plt.title('Area of ArUco Markers vs. Time')
        plt.grid(True)
        plt.show()

class FrameProcessor:
    def __init__(self, cam_calibration):
        self.cam_mat, self.dist_coeff = cam_calibration

    def intrinsic(self, frame):
        return cv2.undistort(frame, self.cam_mat, self.dist_coeff)

    def crop_image(self, img, aruco_found):
        # Implement cropping logic based on ArUco markers
        pass

    def morphologic(self, img):
        # Implement morphological operations
        pass

    def find_balls(self, img_raw, img_cr, output_dir, filename):
        # Implement ball finding logic
        pass

class CSVCombiner:
    def __init__(self, start, step, dir_name):
        self.start = start
        self.step = step
        self.dir_name = dir_name

    def combine_csv_files(self):
        dfs = []
        y = self.start
        for root, dirs, files in os.walk(self.dir_name):
            for file in sorted(files):
                if file.endswith('.csv') and file != 'combined_green_balls.csv':
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    if 'y' in df.columns:
                        pass
                    df['y'] = y
                    dfs.append(df)
            y += self.step

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_csv_path = os.path.join(self.dir_name, 'combined_green_balls.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        return combined_df

class Visualizer:
    @staticmethod
    def visualize_3D_distribution(df, colorscale='Jet'):
        X = df['x'].values
        Y = df['y'].values
        Z = df['z'].values

        fig = go.Figure(data=[go.Mesh3d(x=X, y=Y, z=Z, intensity=Z, colorscale=colorscale)])
        fig.update_layout(
            title="Lechler Nozzle Distribution",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            )
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
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode

    def process_images(self, PRODUCTION_MODE=False):
        import liqdist_archit as ld
        ld.DEBUG_MODE = self.debug_mode
        ld.PRODUCTION_MODE = PRODUCTION_MODE

        if self.debug_mode:
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

            if self.debug_mode:
                print("Image captured")
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "image_captured.png"))

            img_intrinsic = ld.intrinsic(capture, camera_matrix, distortion_coefficients)

            if self.debug_mode:
                print("Image after Undistortion")
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(img_intrinsic, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "image_after_undistortion.png"))

            arucoFound = ld.detect_arucos(capture, camera_matrix, distortion_coefficients)
            if self.debug_mode:
                if arucoFound is not None:
                    print("No of Aruco found: ", len(arucoFound))
                print("Normal image expects 4 aruco detections and live camera for some reason needs 8")
                print("The detected arucos are: ", arucoFound)

            img_cr = ld.crop_image(img_intrinsic, arucoFound)
            if self.debug_mode:
                print("Cropped Images")
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(img_cr, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "cropped_image.png"))

            img_raw = ld.morphologic(img_cr)
            if self.debug_mode:
                print("Image Morphed")
                plt.figure(figsize=(10, 10))
                plt.imshow(img_raw)
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "image_morphed.png"))

            balls_found = ld.find_balls(img_raw, img_cr, output_dir, filename)
            cv2.imwrite(os.path.join(output_dir, "balls_found.png"), balls_found)

if __name__ == "__main__":
    filename = "VidTest2.mp4"
    detector = ArUcoDetector(f'Vids/{filename}', cooldown_time=8)
    detector.detect_aruco_closest_frame()
    
    # Assuming camera calibration is loaded here
    camera_matrix, distortion_coefficients = load_camera_calibration()  # Define this function or load the calibration

    frame_processor = FrameProcessor((camera_matrix, distortion_coefficients))

    combiner = CSVCombiner(start=0, step=1, dir_name="data")
    combined_df = combiner.combine_csv_files()

    visualizer = Visualizer()
    fig_3d = visualizer.visualize_3D_distribution(combined_df)
    fig_3d.show()

    fig_heatmap = visualizer.csv_to_2D_Heatmap(combined_df)
    fig_heatmap.show()

    image_processor = ImageProcessor(debug_mode=True)
    image_processor.process_images(PRODUCTION_MODE=False)
