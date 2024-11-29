import numpy as np
import cv2
from PIL import Image
from glob import glob

from .utils import rectify

def load_endovis_calibration(file_path):
    
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                key, value = map(str.strip, line.split("//")[0].split(':'))
                value = np.array([float(val) for val in value.split(" ")]) if " " in value else float(value)
                data[key.lower()] = value
            except:
                pass

    image_size = (int(data["width"]), int(data["height"]))

    camera_0_matrix = np.array([
        [data["camera-0-f"][0], 0, data["camera-0-c"][0]],
        [0, data["camera-0-f"][1], data["camera-0-c"][1]],
        [0, 0, 1]
    ])

    camera_1_matrix = np.array([
        [data["camera-1-f"][0], 0, data["camera-1-c"][0]],
        [0, data["camera-1-f"][1], data["camera-1-c"][1]],
        [0, 0, 1]
    ])

    Rectify1, Rectify2, Pn1, Pn2, _, _, _ = cv2.stereoRectify(
        camera_0_matrix,
        data["camera-0-k"],
        camera_1_matrix,
        data["camera-1-k"],
        image_size,
        cv2.Rodrigues(data["extrinsic-omega"])[0],
        data["extrinsic-t"],
        0,
        (0, 0)
    )
    
    mapL1, mapL2 = cv2.initUndistortRectifyMap(camera_0_matrix, np.zeros((1,5)), Rectify1, Pn1, image_size, cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(camera_1_matrix, np.zeros((1,5)), Rectify2, Pn2, image_size, cv2.CV_32FC1)
        
    return image_size, mapL1, mapL2, mapR1, mapR2


class EndoVis():
    def __init__(self, directory):
        super().__init__()
        left_frames = sorted(glob(f"{directory}/**/left_frames/*.png", recursive=True))
        right_frames = [path.replace("left_frames", "right_frames") for path in left_frames]
        self.samples = list(zip(left_frames, right_frames))
        calibration_files = glob(f"{directory}/**/camera_calibration.txt", recursive=True)
        self.calibrations = {"/".join(f.split("/")[:-1]): load_endovis_calibration(f) for f in calibration_files}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        left_frame, right_frame = self.samples[index]
        calibration = self.calibrations["/".join(left_frame.split("/")[:-2])]
        left_frame = np.array(Image.open(left_frame))
        right_frame = np.array(Image.open(right_frame))
        left_frame, right_frame = rectify(left_frame, right_frame, calibration)
        return left_frame, right_frame
