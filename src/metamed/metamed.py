import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, ConcatDataset
from torchvision.io import read_image
from typing import Tuple
import imageio
from torchvision.transforms.functional import to_tensor
from random import randint
from math import ceil


SPLITS = [
    "Unlabeled", "UnlabeledPairs", "Training", "Validation", "Testing", "Testing-SCARED", "Testing-SERVCT",
]

class JointTransform():
    def __init__(self, spatial=None, color=None, transform_depth=True):
        self.spatial = spatial
        self.color = color
        self.transform_depth = transform_depth
        
    def __call__(self, image, other_image=None, depth=None):
        if self.spatial != None:
            if depth != None and self.transform_depth:
                if other_image != None:
                    stack = self.spatial(torch.cat([image, other_image, depth], dim=-3))
                    image, other_image, depth = stack[..., :3, :, :], stack[..., 3:6, :, :], stack[..., 6:, :, :]
                else:
                    stack = self.spatial(torch.cat([image, depth], dim=-3))
                    image, depth = stack[..., :3, :, :], stack[..., 3:, :, :]
            elif other_image != None:
                stack = self.spatial(torch.cat([image, other_image], dim=-3))
                image, other_image = stack[..., :3, :, :], stack[..., 3:, :, :]
            else:
                image = self.spatial(image)
        if self.color != None:
            if other_image != None:
                stack = self.color(torch.stack([image, other_image]))
                image, other_image = stack[0], stack[1]
            else:
                image = self.color(image)
                
        if depth != None:
            return image, depth
        elif other_image != None:
            return image, other_image
        else:
            return image


class MetaMED(Dataset):
    def __init__(self, directory, split, spatial_transform=None, color_transform=None, transform_depth=True, stereo=False) -> None:
        super().__init__()
        
        transform = JointTransform(spatial_transform, color_transform, transform_depth)
    
        if split == "Unlabeled":
            self.data = ConcatDataset([
                UnlabeledFrames(f"{directory}/UnlabeledClips/Cholec80", transform, pairs=False),
                UnlabeledFrames(f"{directory}/UnlabeledClips/ROBUST-MIS", transform, pairs=False),
            ])
        elif split == "UnlabeledPairs":
            self.data = ConcatDataset([
                UnlabeledFrames(f"{directory}/UnlabeledClips/Cholec80", transform, pairs=True),
                UnlabeledFrames(f"{directory}/UnlabeledClips/ROBUST-MIS", transform, pairs=True),
            ])
        elif split == "Training":
            self.data = ConcatDataset([
                PsuedoLabeledData(f"{directory}/Training/EndoVis2017", transform, stereo),
                PsuedoLabeledData(f"{directory}/Training/EndoVis2018", transform, stereo),
                PsuedoLabeledData(f"{directory}/Training/KidneyBoundary", transform, stereo),
                PsuedoLabeledData(f"{directory}/Training/StereoMIS", transform, stereo),
            ])
        elif split == "Validation":
            self.data = PsuedoLabeledData(f"{directory}/Validation/Hamlyn", transform, stereo)
        elif split == "Testing":
            self.data = ConcatDataset([
                SCARED(f"{directory}/Testing/SCARED", transform, stereo),
                SERVCT(f"{directory}/Testing/SERV-CT", transform, stereo),
            ])
        elif split == "Testing-SCARED":
            self.data = SCARED(f"{directory}/Testing/SCARED", transform, stereo)
        elif split == "Testing-SERVCT":
            self.data = SERVCT(f"{directory}/Testing/SERV-CT", transform, stereo)
        else:
            raise ValueError(f"Invalid split, should be one of {SPLITS}")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.__getitem__(index)
    
    
class UnlabeledFrames(Dataset):
    def __init__(self, directory, transform, pairs=False, max_time_step=0.2):
        super().__init__()
        self.clips = sorted(glob(f"{directory}/**/clip-*", recursive=True))
        self.transform = transform
        self.pairs = pairs
        self.max_frame_step = ceil(max_time_step * 25.0)
        
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
        
        images = glob(f"{self.clips[index]}/*.png")
        
        if self.pairs:
            step = randint(1, self.max_frame_step)
            first_index = randint(0, len(images)-step-1)
            second_index = first_index + step
            first_image = read_image(images[first_index])[:3] / 255.0
            second_image = read_image(images[second_index])[:3] / 255.0
            if self.transform is not None:
                first_image, second_image = self.transform(first_image, other_image=second_image)
            return first_image, second_image
        else:
            index = randint(0, len(images)-1)
            image = read_image(images[index])[:3] / 255.0
            if self.transform is not None:
                image = self.transform(image)
            return image


class PsuedoLabeledData(Dataset):
    def __init__(self, directory, transform, stereo=False) -> None:
        super().__init__()
        left_frames = sorted([path for path in glob(f"{directory}/**/*.png", recursive=True) if "left" in path])
        right_frames = [path.replace("left", "right") for path in left_frames]
        disparities = [path.replace("left", "disparity").replace(".png", ".npy") for path in left_frames]
        self.samples = list(zip(left_frames, right_frames, disparities))
        self.transform = transform
        self.stereo = stereo
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        left_frame_path, right_frame_path, dispartiy_path = self.samples[index]
        
        left_frame = read_image(left_frame_path)[:3] / 255.0
        if self.stereo:
            right_frame = read_image(right_frame_path)[:3] / 255.0
        depth = torch.from_numpy(np.load(dispartiy_path)).permute(2, 0, 1)
        
        if self.stereo:
            if self.transform != None:
                left_frame, right_frame, depth = self.transform(left_frame, other_image=right_frame, depth=depth)
            return left_frame, right_frame, depth
        else:
            if self.transform != None:
                left_frame, depth = self.transform(left_frame, depth=depth)
            return left_frame, depth
            



class SCARED(Dataset):
    def __init__(self, directory, transform, stereo=False) -> None:
        super().__init__()
        left_frames = sorted(glob(f"{directory}/dataset*/keyframe_*/Left_Image.png", recursive=True))
        right_frames = [path.replace("Left", "Right") for path in left_frames]
        depths = [path.replace("Left_Image.png", "left_depth_map.tiff") for path in left_frames]
        self.samples = list(zip(left_frames, right_frames, depths))
        self.transform = transform
        self.stereo = stereo
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        left_frame_path, right_frame_path, depth_path = self.samples[index]
        
        left_frame = read_image(left_frame_path)[:3] / 255.0
        if self.stereo:
            right_frame = read_image(right_frame_path)[:3] / 255.0
            
        depth = -1.0 / to_tensor(imageio.read(depth_path).get_data(0)).float()[2:3]
        sorted = torch.sort(depth[~torch.isnan(depth)]).values
        mi, ma = sorted[:2000].mean(), sorted[-2000:].mean()
        depth = torch.clamp(depth, mi, ma)
       
        if self.stereo:
            if self.transform is not None:
                left_frame, right_frame, depth = self.transform(left_frame, other_image=right_frame, depth=depth)
            return left_frame, right_frame, depth
        else:
            if self.transform is not None:
                left_frame, depth = self.transform(left_frame, depth=depth)
            return left_frame, depth



class SERVCT(Dataset):
    def __init__(self, directory, transform, stereo=False) -> None:
        super().__init__()
        left_frames = sorted(glob(f"{directory}/*/Left_rectified/*.png", recursive=True))
        right_frames = [path.replace("Left", "Right") for path in left_frames]
        depths = [path.replace("Left_rectified", "Ground_truth_CT/DepthL") for path in left_frames]
        self.samples = list(zip(left_frames, right_frames, depths))
        self.transform = transform
        self.stereo = stereo
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        left_frame_path, right_frame_path, depth_path = self.samples[index]
        
        left_frame = read_image(left_frame_path)[:3] / 255.0
        if self.stereo:
            right_frame = read_image(right_frame_path)[:3] / 255.0
            
        depth = -1.0 / to_tensor(imageio.imread(depth_path).astype(float)).float()[:1]
       
        if self.stereo:
            if self.transform is not None:
                left_frame, right_frame, depth = self.transform(left_frame, other_image=right_frame, depth=depth)
            return left_frame, right_frame, depth
        else:
            if self.transform is not None:
                left_frame, depth = self.transform(left_frame, depth=depth)
            return left_frame, depth
