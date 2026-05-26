import torch
import numpy as np
from model import LipNet
from inference import extract_lips_from_video, greedy_decode

def predict_video(video_path, weights_path="models/best_lipnet.pth", window_size=29):
    """
    Extracts lips from a raw video, runs inference, and returns the decoded prediction text.
    
    Args:
        video_path: Path to the raw input video (.mpg, .mp4, etc.)
        weights_path: Path to the saved PyTorch model weights (.pth)
        window_size: Input temporal frame length expected by the model (default 29)
        
    Returns:
        prediction: Predicted text string
    """
    # Step 1: Extract lip region frames (T, 80, 80)
    frames = extract_lips_from_video(video_path)
    if frames is None:
        raise ValueError("Could not extract frames or detect lips in the video.")
    
    T = frames.shape[0]
    
    # Step 2: Downsample temporal length to window_size (29)
    if T != window_size:
        indices = np.round(np.linspace(0, T - 1, window_size)).astype(int)
        frames = frames[indices]
        
    # Step 3: Normalize and format to shape: (1, 29, 1, 80, 80)
    frames_norm = frames.astype(np.float32) / 255.0
    frames_norm = np.expand_dims(frames_norm, axis=1)    # (29, 1, 80, 80)
    frames_norm = np.expand_dims(frames_norm, axis=0)    # (1, 29, 1, 80, 80)
    
    input_tensor = torch.tensor(frames_norm, dtype=torch.float32)
    
    # Step 4: Load model
    device = torch.device("cpu")
    model = LipNet(vocab_size=28).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Step 5: Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = greedy_decode(outputs)
        
    return prediction

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_video>")
    else:
        video_path = sys.argv[1]
        try:
            pred = predict_video(video_path)
            print(f"Prediction: {pred}")
        except Exception as e:
            print(f"Error: {e}")
