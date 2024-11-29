import os
import torch
import numpy as np
from torchvision.transforms.functional import resize, normalize
from torchvision.io import write_png

from src.training.opticalflow import OpticalFlow

def calculate_stereo_depth(samples, sub_dirs, target_directory):

    optical_flow = OpticalFlow(mask_mode="flow").cuda()
    
    previous_sub_dir = None

    for (left_frame, right_frame), sub_dir in zip(samples, sub_dirs):

        left_frame = torch.from_numpy(left_frame).permute(2, 0, 1).cuda()
        right_frame = torch.from_numpy(right_frame).permute(2, 0, 1).cuda()
        
        mask = (torch.sum(left_frame[0], dim=0) + torch.sum(right_frame[0], dim=0)) > 0
        first = torch.argwhere(mask.cumsum(0) == 1)[0]
        last = mask.size(0) - torch.argwhere(mask.flip(0).cumsum(0) == 1)[0]
        left_frame = left_frame[:, :, first:last]
        right_frame = right_frame[:, :, first:last]
        
        left_frame = resize(left_frame, (384, 384))
        right_frame = resize(right_frame, (384, 384))
        left_frame_normed = normalize(left_frame / 255.0, 0.5, 0.5)
        right_frame_normed = normalize(right_frame / 255.0, 0.5, 0.5)

        with torch.no_grad():
            flow, mask = optical_flow.get_flows(left_frame_normed.unsqueeze(0), right_frame_normed.unsqueeze(0), get_mask=True)

        disp = torch.where(mask[0], flow[0, 0:1], torch.nan)
        
        if torch.mean(mask.float()).item() > 0.2:
            
            if sub_dir != previous_sub_dir:
                os.makedirs(f"{target_directory}/{sub_dir}", exist_ok=True)
                previous_sub_dir = sub_dir
                sample_count = 0
            else:
                sample_count += 1
                
            write_png(left_frame.cpu(), f"{target_directory}/{sub_dir}/{sample_count:04d}-left.png")
            write_png(right_frame.cpu(), f"{target_directory}/{sub_dir}/{sample_count:04d}-right.png")
            np.save(f"{target_directory}/{sub_dir}/{sample_count:04d}-disparity.npy", disp.permute(1, 2, 0).cpu().numpy())
