import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Read Paths
INPUT_PATH = r"G:\Drive'ım\ain313as4\processed_features"

# Save
OUTPUT_PATH = r"G:\Drive'ım\ain313as4\ready_data"

# Actions
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

# Fixed sequence length
MAX_TIMESTEPS = 50 

def load_and_preprocess_data():
    if not os.path.exists(INPUT_PATH):
        print(f"Couldn't find dataset: {INPUT_PATH}")
        print("Run openpose.py to extract features first")
        return

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    all_features = []
    all_labels = []
    video_lengths = []

    print(f"Reading datas: {INPUT_PATH}")

    for action in ACTIONS:
        action_dir = os.path.join(INPUT_PATH, action)
        npz_files = glob.glob(os.path.join(action_dir, "*.npz"))

        # Process each .npz file
        for f_path in npz_files:
            try:
                data = np.load(f_path, allow_pickle=True)
                poses = data['normalized_keypoints'] 
                
                if len(poses) == 0: continue 
                
                # Flatten: (N, 25, 3) -> (N, 75)
                flattened_poses = poses.reshape(poses.shape[0], -1)
                
                # Frame step 2 
                all_features.append(flattened_poses)
                all_labels.append(action)
                video_lengths.append(len(flattened_poses))
            except Exception as e:
                print(f"Hata: {f_path} - {e}")

    # Video length analysis
    video_lengths = np.array(video_lengths)
    print("\nVideo length analysis after frame step 2:")
    if len(video_lengths) > 0:
        print(f"Min: {np.min(video_lengths)}")
        print(f"Max: {np.max(video_lengths)}")
        print(f"Avg: {np.mean(video_lengths):.2f}")
    
    # Padding / Truncating
    X = []
    for seq in all_features:
        seq_len = len(seq)
        feature_dim = seq.shape[1] 
        
        if seq_len < MAX_TIMESTEPS:
            # Padding 
            padding = np.zeros((MAX_TIMESTEPS - seq_len, feature_dim))
            new_seq = np.vstack((seq, padding))
        else:
            # Truncating
            new_seq = seq[:MAX_TIMESTEPS, :]
            
        X.append(new_seq)

    # Transform to numpy array
    X = np.array(X) 
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nData preprocessing completed:")
    print(f"X_train: {X_train.shape}") 
    print(f"X_test:  {X_test.shape}")

    # Save prepared data
    np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_PATH, "label_classes.npy"), le.classes_)
    
    print(f"Saved on{OUTPUT_PATH}")

# run the data loading and preprocessing
load_and_preprocess_data()