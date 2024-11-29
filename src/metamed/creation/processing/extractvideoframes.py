import cv2
import os
import torch
import numpy as np
from torchcontentarea import estimate_area, get_crop


def calculate_crop(frames):
    frames = torch.from_numpy(np.stack(frames[::50]))
    frames = frames.permute(0, 3, 1, 2).flip(1).cuda()
    areas = estimate_area(frames)[:, :3]
    mean, std = areas.mean(dim=0), areas.std(dim=0)
    if torch.sum(std) > 0:
        mask = (torch.abs(areas - mean[None, :]) > 3 * std).sum(dim=1) == 0
        areas = areas[mask, :]
    area = torch.mean(areas, dim=0)
    try:
        crop = get_crop(area, frames.shape[-2:], bias=-10)
    except Exception as e:
        print(areas)
        print(mean, std)
        print(mask)
        print(area)
        raise e
    return crop


def write_frames(frames, directory):
    
    if os.path.exists(directory) and len(os.listdir(directory)) == 250:
        return
    
    os.makedirs(directory, exist_ok=True)
 
    t, b, l, r = calculate_crop(frames)
    
    for i, frame in enumerate(frames):
        frame = frame[t:b, l:r]
        frame_filename = f"{directory}/frame-{i:03d}.png"
        cv2.imwrite(frame_filename, frame)
                    

def extract_video_frames(video_files, sub_dirs, target_directory, clip_seconds):
    
    
    for video_file, sub_dir in zip(video_files, sub_dirs):
        
        print(video_file)
        
        clip_count = 0
    
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_file}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # ROBUST-MIS uses blue to indicate out of body frames
            average_color = frame.mean(axis=0).mean(axis=0)
            if average_color[0] > 250 and (average_color[1:] < 10).all():
                frames = []
                continue
        
            frames.append(frame)
            if len(frames) / fps >= clip_seconds:
                write_frames(frames, f"{target_directory}/{sub_dir}/clip-{clip_count:05d}")
                frames = []
                clip_count += 1
        
        cap.release()
    
    print("Frame extraction completed.")
