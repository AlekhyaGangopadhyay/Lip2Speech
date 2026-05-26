import os
import cv2
import torch
import numpy as np
import argparse
from model import LipNet
from dataset import decode_text

# MediaPipe lip landmark indices (outer lip loop)
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95]

def extract_lips_from_video(video_path):
    # Initialize face mesh inside the function
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

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
            lip_coords = []
            for idx in LIP_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                lip_coords.append((lm.x * w, lm.y * h))
            
            lip_coords = np.array(lip_coords)
            center_x = np.mean(lip_coords[:, 0])
            center_y = np.mean(lip_coords[:, 1])
            last_center = (center_x, center_y)
        elif last_center is not None:
            center_x, center_y = last_center
        else:
            center_x, center_y = w / 2, h / 2

        center_x = max(40, min(w - 40, center_x))
        center_y = max(40, min(h - 40, center_y))

        x1 = int(center_x - 40)
        y1 = int(center_y - 40)
        
        crop = gray[y1:y1+80, x1:x1+80]
        frames.append(crop)

    cap.release()
    face_mesh.close()

    if len(frames) == 0:
        return None

    return np.stack(frames, axis=0)

def greedy_decode(outputs):
    arg_maxes = torch.argmax(outputs, dim=-1)  # (Batch, T)
    indices = arg_maxes[0]  # First element in batch
    
    collapsed = []
    prev = -1
    for idx in indices:
        idx = idx.item()
        if idx != prev:
            if idx != 0:  # skip blank token
                collapsed.append(idx)
            prev = idx
    return decode_text(collapsed)

def main():
    parser = argparse.ArgumentParser(description="LipNet Inference Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to raw input video (.mpg / .mp4)")
    parser.add_argument("--weights", type=str, default="models/best_lipnet.pth", help="Path to saved model weights")
    parser.add_argument("--window_size", type=int, default=29, help="Number of frames to feed into the model")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}. Please train the model first.")
        return

    print("Step 1: Extracting lip region from video...")
    raw_frames = extract_lips_from_video(args.video)
    if raw_frames is None:
        print("Error: Could not extract frames from video.")
        return

    T = raw_frames.shape[0]
    print(f"Extracted {T} raw frames.")

    # Step 2: Downsample to window_size frames (default 29) to match model expectations
    if T != args.window_size:
        print(f"Downsampling temporal length from {T} to {args.window_size} frames...")
        indices = np.round(np.linspace(0, T - 1, args.window_size)).astype(int)
        frames = raw_frames[indices]
    else:
        frames = raw_frames

    # Normalize to [0, 1] and add Batch and Channel dimensions: (29, 80, 80) -> (1, 29, 1, 80, 80)
    frames_norm = frames.astype(np.float32) / 255.0
    frames_norm = np.expand_dims(frames_norm, axis=1)    # (29, 1, 80, 80)
    frames_norm = np.expand_dims(frames_norm, axis=0)    # (1, 29, 1, 80, 80)
    
    input_tensor = torch.tensor(frames_norm, dtype=torch.float32)

    print("Step 3: Loading model...")
    device = torch.device("cpu")
    model = LipNet(vocab_size=28).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    print("Step 4: Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)                    # (1, 29, VocabSize)
        predicted_text = greedy_decode(outputs)

    print("\n====================================")
    print(f"PREDICTED TEXT: {predicted_text}")
    print("====================================\n")

if __name__ == "__main__":
    main()
