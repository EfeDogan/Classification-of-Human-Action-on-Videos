import cv2
import numpy as np
import os
import glob
import json
import subprocess
import shutil
import time

# Openpose path
OPENPOSE_DIR = r"C:\openpose_gpu"  
OPENPOSE_EXE = os.path.join(OPENPOSE_DIR, "bin", "OpenPoseDemo.exe")

# Google Drive root path
DRIVE_ROOT = r"G:\Drive'ım\ain313as4" 

# Video dataset path
DATASET_PATH = os.path.join(DRIVE_ROOT, "video_dataset", "video_dataset")

# Processed features output path
OUTPUT_PATH = os.path.join(DRIVE_ROOT, "processed_features")

# Actions
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

# Main processing function
def process_dataset_drive_final():
    
    # Check paths
    if not os.path.exists(OPENPOSE_EXE):
        print(f"Couldn't find openpose: {OPENPOSE_EXE}")
        return
    
    if not os.path.exists(DATASET_PATH):
        print(f"Couldn't find video folder: {DATASET_PATH}")
        return

    print("="*60)
    print(f"Input Path: {DATASET_PATH}")
    print(f"Output Path:   {OUTPUT_PATH}")
    print("="*60)

    start_time_total = time.time()

    # processing for each action 
    for action in ACTIONS:
        action_path = os.path.join(DATASET_PATH, action)
        save_dir = os.path.join(OUTPUT_PATH, action)
        os.makedirs(save_dir, exist_ok=True)

        # temp json buffer
        temp_json_dir = os.path.join(OPENPOSE_DIR, "_temp_json_buffer") 

        # Find videos
        video_files = sorted(glob.glob(os.path.join(action_path, "*.avi")))
        
        if not video_files:
            print(f"\nWarning: {action.upper()} no video on folder! ({action_path})")
            continue

        print(f"\nCategory: {action.upper()} ({len(video_files)} video)")

        # Process each video
        for i, video_path in enumerate(video_files):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(save_dir, video_name + ".npz")

            # Clear temp json dir
            if os.path.exists(temp_json_dir): shutil.rmtree(temp_json_dir)
            os.makedirs(temp_json_dir)

            # OpenPose command
            command = [
                    OPENPOSE_EXE,
                    "--video", video_path,
                    "--write_json", temp_json_dir,
                    "--model_pose", "BODY_25",
                    "--display", "0",
                    "--render_pose", "0",
                    
                    # Quality setting: 128 
                    "--net_resolution", "-1x240",
                    
                    # Noise reduction
                    "--number_people_max", "1",
                    
                    # Disable blending
                    "--disable_blending",
                    
                    # Speed: frame step 2
                    "--frame_step", "2",
                    
                    # GPU settings
                    "--num_gpu", "1",
                    "--num_gpu_start", "0"
            ]

            print(f"   [{i+1}/{len(video_files)}] {video_name} ...", end="")
            video_start_time = time.time()

            try:
                subprocess.run(
                    command, 
                    cwd=OPENPOSE_DIR, 
                    check=True,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError:
                print("\nError: OpenPose couldn't run.")
                continue

            # JSON -> NPZ
            json_files = sorted(glob.glob(os.path.join(temp_json_dir, "*.json")))
            raw_keypoints, norm_keypoints, frame_indices = [], [], []

            # Get video dimensions
            cap = cv2.VideoCapture(video_path)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if cap.isOpened() else 160
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if cap.isOpened() else 120
            cap.release()


            # Process each JSON to extract keypoints
            for idx, j_file in enumerate(json_files):
                try:
                    with open(j_file, "r") as f:
                        data = json.load(f)
                    if data["people"]:
                        pose = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(25, 3)
                        norm_pose = pose.copy()
                        norm_pose[:, 0] /= w
                        norm_pose[:, 1] /= h
                        raw_keypoints.append(pose)
                        norm_keypoints.append(norm_pose)
                        frame_indices.append(idx) 
                except: pass

            # Save to .npz
            if raw_keypoints:
                np.savez(save_path, 
                         keypoints=np.array(raw_keypoints), 
                         normalized_keypoints=np.array(norm_keypoints), 
                         frame_indices=np.array(frame_indices), 
                         label=action, video_path=video_path)
                print(f" ✅ ({time.time() - video_start_time:.2f}s)")
            else:
                print("Empty. No keypoints extracted.")

        if os.path.exists(temp_json_dir): shutil.rmtree(temp_json_dir)

    print(f"\nAll done, folders are here: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_dataset_drive_final()