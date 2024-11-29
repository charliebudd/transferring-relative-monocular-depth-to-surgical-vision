import numpy as np
from PIL import Image
from glob import glob
import cv2
import os
from itertools import chain

from .utils import rectify

def load_hamlyn_calibration(calibration_dir):
    
    def load(file):
        with open(file) as f:
            data = [list(map(float, f.readline().split(None))) for _ in range(4)]
        return np.array(data[:3]), np.array(data[3])
    
    image_size = Image.open(f"{calibration_dir}/left/0001.png").size
    
    rotation_matrix, translation_vector = load(f"{calibration_dir}/camera_extrinsic.txt")
    left_camera_matrix, left_distortion_coeffs = load(f"{calibration_dir}/Left_Camera_Calibration_Intrinsic.txt")
    right_camera_matrix, right_distortion_coeffs = load(f"{calibration_dir}/Right_Camera_Calibration_Intrinsic.txt")

    # Perform stereo rectification
    Rectify1, Rectify2, Pn1, Pn2, _, _, _ = cv2.stereoRectify(
        left_camera_matrix,
        left_distortion_coeffs,
        right_camera_matrix,
        right_distortion_coeffs,
        image_size,
        rotation_matrix,
        translation_vector,
        0,
        (0, 0),
    )

    # Initialize undistort rectify maps
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        left_camera_matrix,
        left_distortion_coeffs,
        Rectify1,
        Pn1,
        image_size,
        cv2.CV_32FC1
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        right_camera_matrix,
        right_distortion_coeffs,
        Rectify2,
        Pn2,
        image_size,
        cv2.CV_32FC1
    )

    return image_size, mapL1, mapL2, mapR1, mapR2

# class Hamlyn():
#     def __init__(self, directory):
#         super().__init__()
#         left_frames = sorted(glob(f"{directory}/dataset*/left/*.png"))
#         right_frames = [path.replace("left", "right") for path in left_frames]
#         self.samples = list(zip(left_frames, right_frames))
#         dataset_dirs = glob(f"{directory}/dataset*")
#         self.calibrations = {dir: load_hamlyn_calibration(dir) for dir in dataset_dirs}
        
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         left_frame, right_frame = self.samples[index]
#         calibration = self.calibrations["/".join(left_frame.split("/")[:-2])]
#         left_frame = np.array(Image.open(left_frame))
#         right_frame = np.array(Image.open(right_frame))
#         left_frame, right_frame = rectify(left_frame, right_frame, calibration)
#         return left_frame, right_frame


class Hamlyn():
    def __init__(self, directory):
        super().__init__()
        self.videos = [HamlynVideo(dir) for dir in glob(f"{directory}/dataset*")]
        
    def __len__(self):
        return sum([video.sample_count for video in self.videos])
    
    def __iter__(self):
        for video in self.videos:
            yield from video


class HamlynVideo():
    def __init__(self, directory):
        self.stereo_path = f"{directory}/stereo.avi"
        self.left_path = f"{directory}/left.avi"
        self.right_path = f"{directory}/right.avi"
        
        self.calibration = load_hamlyn_calibration(directory)
        
        if os.path.exists(self.stereo_path):
            self.has_stereo_file = True
            main_video = cv2.VideoCapture(self.stereo_path)
        else:
            self.has_stereo_file = False
            main_video = cv2.VideoCapture(self.left_path)
            
        frame_count = int(main_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(main_video.get(cv2.CAP_PROP_FPS) + 0.1)
        self.sample_count = int(frame_count // self.fps)
        
        self.count = 0
        
        main_video.release()
    
    def __iter__(self):
        
        print("################")
        print(self.sample_count, self.has_stereo_file)
        
        if self.has_stereo_file:
            self.stereo_video = cv2.VideoCapture(self.stereo_path)
        else:
            self.left_video = cv2.VideoCapture(self.left_path)
            self.right_video = cv2.VideoCapture(self.right_path)
        
        return self
    
    def __next__(self):
        
        if self.has_stereo_file:
            for _ in range(self.fps):
                ret, frame = self.stereo_video.read()
                if not ret:
                    self.stereo_video.release()
                    print(self.count)
                    raise StopIteration()
            width = frame.shape[1]
            frame = np.flip(frame, axis=2)
            left = frame[:, :width//2]
            right = frame[:, width//2:]
        else:
            for _ in range(self.fps):
                ret, left = self.left_video.read()
                if not ret:
                    self.left_video.release()
                    self.right_video.release()
                    print(self.count)
                    raise StopIteration()
                ret, right = self.right_video.read()
                if not ret:
                    self.left_video.release()
                    self.right_video.release()
                    print(self.count)
                    raise StopIteration()
            left = np.flip(left, axis=2)
            right = np.flip(right, axis=2)
            
        self.count += 1
        left, right = rectify(left, right, self.calibration)
        
        return left, right
        