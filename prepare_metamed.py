import argparse
import shutil
from itertools import chain
from glob import glob
from src.metamed.creation.loaders import EndoVis, Hamlyn, StereoMIS
from src.metamed.creation.processing import calculate_stereo_depth, extract_video_frames

def main(args):
    
    # ===================================
    # Unlabelled training data...

    robustmis_videos = glob(f"{args.dataset_directory}/ROBUST-MIS/Raw data/*/*/*.avi")
    assert len(robustmis_videos) == 30, "Could not find all ROBUST-MIS videos."
    robustmis_sub_dirs = [f"ROBUST-MIS/{f.split('/')[-3]}/{f.split('/')[-2]}" for f in robustmis_videos]

    cholec80_videos = glob(f"{args.dataset_directory}/Cholec80/videos/*.mp4")
    assert len(cholec80_videos) == 80, "Could not find all Cholec80 videos."
    cholec80_sub_dirs = [f"Cholec80/{f.split('/')[-1].split('.')[0]}" for f in cholec80_videos][::-1]

    all_videos = robustmis_videos + cholec80_videos
    all_sub_dirs = robustmis_sub_dirs + cholec80_sub_dirs

    extract_video_frames(all_videos, all_sub_dirs, "Meta-MED/UnlabeledClips", clip_seconds=10.0)

    # ===================================
    # Pseudo labelled training and validation data...

    endovis2017_samples = EndoVis(f"{args.dataset_directory}/EndoVis2017")
    assert len(endovis2017_samples) == 3000, "Could not find all EndoVis2017 samples."
    
    endovis2018_samples = EndoVis(f"{args.dataset_directory}/EndoVis2018")
    assert len(endovis2018_samples) == 3234, "Could not find all EndoVis2018 samples."

    kidneyboundary_samples = EndoVis(f"{args.dataset_directory}/KidneyBoundary")
    assert len(kidneyboundary_samples) == 1500, "Could not find all KidneyBoundary samples."
    
    stereomis_samples = StereoMIS(f"{args.dataset_directory}/StereoMIS")
    assert len(stereomis_samples) == 2702, "Could not find all StereoMIS samples."
    
    hamlyn_samples = Hamlyn(f"{args.dataset_directory}/Hamlyn")
    assert len(hamlyn_samples) == 1819, "Could not find all Hamlyn samples."
    
    all_samples = chain(
        endovis2017_samples,
        endovis2018_samples,
        kidneyboundary_samples,
        stereomis_samples,
        hamlyn_samples,
    )
    
    all_sub_dirs =chain(
        len(endovis2017_samples) * ["Training/EndoVis2017"],
        len(endovis2018_samples) * ["Training/EndoVis2018"],
        len(kidneyboundary_samples) * ["Training/KidneyBoundary"],
        len(stereomis_samples) * ["Training/StereoMIS"],
        len(hamlyn_samples) * ["Validation/Hamlyn"],
    )

    calculate_stereo_depth(all_samples, all_sub_dirs, "Meta-MED")
    
    # ===================================
    # Evaluation data...

    scared_samples = glob(f"{args.dataset_directory}/SCARED/dataset_*/keyframe_*/left_depth_map.tiff")
    assert len(scared_samples) == 45, "Could not find all SCARED samples."
    shutil.copytree(f"{args.dataset_directory}/SCARED", "Meta-MED/Testing/SCARED", ignore=shutil.ignore_patterns("*.obj", "*.mp4", "*.tar.gz"))

    servct_samples = glob(f"{args.dataset_directory}/SERV-CT/Experiment_*/Ground_truth_CT/DepthL/*.png")
    assert len(servct_samples) == 16, "Could not find all SERV-CT samples."
    shutil.copytree(f"{args.dataset_directory}/SERV-CT", "Meta-MED/Testing/SERV-CT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", required=True, help="Directory containing all input datasets")
    args = parser.parse_args()
    main(args)
