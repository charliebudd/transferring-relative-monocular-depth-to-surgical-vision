
from glob import glob
import cv2
from PIL import Image
import numpy as np

import configparser


from .utils import rectify


def load_stereomis_calibration(file_path):

    # Load the calibration parameters from the INI file
    config = configparser.ConfigParser()
    config.read(file_path)

    # Extract calibration parameters for the left camera
    camera_0_matrix = np.array([
        [float(config["StereoLeft"]["fc_x"]), 0, float(config["StereoLeft"]["cc_x"])],
        [0, float(config["StereoLeft"]["fc_y"]), float(config["StereoLeft"]["cc_y"])],
        [0, 0, 1]
    ])

    # Extract calibration parameters for the right camera
    camera_1_matrix = np.array([
        [float(config["StereoRight"]["fc_x"]), 0, float(config["StereoRight"]["cc_x"])],
        [0, float(config["StereoRight"]["fc_y"]), float(config["StereoRight"]["cc_y"])],
        [0, 0, 1]
    ])

    data = {
        "camera-0-k": np.array([
            float(config["StereoLeft"]["kc_0"]),
            float(config["StereoLeft"]["kc_1"]),
            float(config["StereoLeft"]["kc_2"]),
            float(config["StereoLeft"]["kc_3"]),
            float(config["StereoLeft"]["kc_4"])
        ]),
        "camera-1-k": np.array([
            float(config["StereoRight"]["kc_0"]),
            float(config["StereoRight"]["kc_1"]),
            float(config["StereoRight"]["kc_2"]),
            float(config["StereoRight"]["kc_3"]),
            float(config["StereoRight"]["kc_4"])
        ]),
        "extrinsic-omega": np.array([
            [float(config["StereoLeft"]["R_0"]), float(config["StereoLeft"]["R_1"]), float(config["StereoLeft"]["R_2"])],
            [float(config["StereoLeft"]["R_3"]), float(config["StereoLeft"]["R_4"]), float(config["StereoLeft"]["R_5"])],
            [float(config["StereoLeft"]["R_6"]), float(config["StereoLeft"]["R_7"]), float(config["StereoLeft"]["R_8"])]
        ]),
        "extrinsic-t": np.array([
            float(config["StereoRight"]["T_0"]),
            float(config["StereoRight"]["T_1"]),
            float(config["StereoRight"]["T_2"])
        ])
    }

    # Continue with the rest of the code
    image_size = (int(config["StereoLeft"]["res_x"]), int(config["StereoLeft"]["res_y"]))

    Rectify1, Rectify2, Pn1, Pn2, _, _, _ = cv2.stereoRectify(
        camera_0_matrix,
        data["camera-0-k"],
        camera_1_matrix,
        data["camera-1-k"],
        image_size,
        data["extrinsic-omega"],
        data["extrinsic-t"],
        0,
        (0, 0)
    )
    
    mapL1, mapL2 = cv2.initUndistortRectifyMap(camera_0_matrix, np.zeros((1,5)), Rectify1, Pn1, image_size, cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(camera_1_matrix, np.zeros((1,5)), Rectify2, Pn2, image_size, cv2.CV_32FC1)

    return image_size, mapL1, mapL2, mapR1, mapR2


class StereoMIS():
    def __init__(self, directory):
        super().__init__()
        left_frames = sorted(glob(f"{directory}/*/*/*_left.png"))
        right_frames = [path.replace("_left.png", "_right.png") for path in left_frames]
        self.samples = list(zip(left_frames, right_frames))
        calibration_files = glob(f"{directory}/*/StereoCalibration.ini")
        self.calibrations = {"/".join(f.split("/")[:-1]): load_stereomis_calibration(f) for f in calibration_files}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        left_frame, right_frame = self.samples[index]
        calibration = self.calibrations["/".join(left_frame.split("/")[:-2])]
        left_frame = np.array(Image.open(left_frame))
        right_frame = np.array(Image.open(right_frame))
        left_frame, right_frame = rectify(left_frame, right_frame, calibration)
        return left_frame, right_frame
