import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
from tqdm import tqdm

# MediaPipe lip landmark indices (outer lip loop)
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95]

def extract_lips_from_video(video_path, face_mesh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frames = []
    last_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face mesh
        results = face_mesh.process(rgb_frame)

        center_x, center_y = None, None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Get coords of lip landmarks
            lip_coords = []
            for idx in LIP_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                lip_coords.append((lm.x * w, lm.y * h))
            
            lip_coords = np.array(lip_coords)
            # Find center of lips
            center_x = np.mean(lip_coords[:, 0])
            center_y = np.mean(lip_coords[:, 1])
            last_center = (center_x, center_y)
        elif last_center is not None:
            # Fallback to last known center
            center_x, center_y = last_center
        else:
            # Absolute fallback to center of the frame
            center_x, center_y = w / 2, h / 2

        # Bounding box crop 80x80 around center, clamped to frame boundary
        center_x = max(40, min(w - 40, center_x))
        center_y = max(40, min(h - 40, center_y))

        x1 = int(center_x - 40)
        y1 = int(center_y - 40)
        
        crop = gray[y1:y1+80, x1:x1+80]
        frames.append(crop)

    cap.release()

    if len(frames) == 0:
        return None

    # Stack to numpy array of shape (T, 80, 80)
    return np.stack(frames, axis=0)

def main():
    parser = argparse.ArgumentParser(description="GRID Corpus Lip Extraction Preprocessing")
    parser.add_argument("--video_dir", type=str, default="d:/Leep2Speech/grid_corpus/s1/video", help="Directory containing raw mpg videos")
    parser.add_argument("--align_dir", type=str, default="d:/Leep2Speech/grid_corpus/s1/align", help="Directory containing alignment transcripts")
    parser.add_argument("--output_dir", type=str, default="d:/Leep2Speech/preprocessed_data", help="Output directory for cropped lip files")
    parser.add_argument("--num_files", type=str, default="100", help="Number of files to process (specify integer, or 'all')")
    args = parser.parse_args()

    video_dir = args.video_dir
    align_dir = args.align_dir
    output_dir = args.output_dir
    
    os.makedirs(os.path.join(output_dir, "lip_tensors"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "transcripts"), exist_ok=True)

    print("Initializing MediaPipe Face Mesh...")
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Get list of video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mpg")]
    video_files.sort()

    if args.num_files.lower() != "all":
        try:
            num_to_process = int(args.num_files)
            video_files = video_files[:num_to_process]
        except ValueError:
            print(f"Invalid num_files value: {args.num_files}. Processing all files.")

    print(f"Processing {len(video_files)} files...")

    success_count = 0
    for v_file in tqdm(video_files):
        video_path = os.path.join(video_dir, v_file)
        base_name = os.path.splitext(v_file)[0]
        
        # Check alignment file
        align_path = os.path.join(align_dir, f"{base_name}.align")
        if not os.path.exists(align_path):
            print(f"Alignment file not found for {v_file}, skipping.")
            continue

        # Extract lips
        lip_tensor = extract_lips_from_video(video_path, face_mesh)
        if lip_tensor is None:
            print(f"Failed to extract lips from {v_file}, skipping.")
            continue

        # Save lip tensor (T, 80, 80)
        output_tensor_path = os.path.join(output_dir, "lip_tensors", f"{base_name}.npy")
        np.save(output_tensor_path, lip_tensor)

        # Copy/Save transcription metadata
        # Parse words from align file
        words = []
        with open(align_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    # We ignore silence tokens if desired, but let's keep all words and just skip 'sil'
                    if word not in ["sil", "sp"]:
                        words.append(word)
        
        sentence = " ".join(words)
        output_txt_path = os.path.join(output_dir, "transcripts", f"{base_name}.txt")
        with open(output_txt_path, "w") as f:
            f.write(sentence)

        success_count += 1

    print(f"Preprocessing completed successfully for {success_count}/{len(video_files)} files.")

if __name__ == "__main__":
    main()
