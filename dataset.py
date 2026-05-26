import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# GRID Corpus character mapping: 0 -> CTC blank, 1 -> space, 2..27 -> a..z
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz"
char_to_num = {char: idx + 2 for idx, char in enumerate(VOCAB_CHARS)}
char_to_num[" "] = 1
char_to_num["_"] = 0

num_to_char = {idx: char for char, idx in char_to_num.items()}

def encode_text(text):
    return [char_to_num[char] for char in text.lower() if char in char_to_num]

def decode_text(indices):
    return "".join([num_to_char[idx] for idx in indices if idx in num_to_char and idx != 0])

class LipReadingDataset(Dataset):
    def __init__(self, output_dir="d:/Leep2Speech/preprocessed_data", split="train", val_ratio=0.2, 
                 window_size=29, window_mode="subsample", random_seed=42):
        """
        Args:
            output_dir: Directory containing preprocessed tensors and transcripts
            split: "train" or "val"
            val_ratio: Ratio of validation samples
            window_size: Size of frame window (e.g. 29 frames)
            window_mode: "subsample" (strided frames of entire video) or "slice" (contiguous crop)
            random_seed: Seed for split reproducibility
        """
        self.output_dir = output_dir
        self.window_size = window_size
        self.window_mode = window_mode

        tensor_dir = os.path.join(output_dir, "lip_tensors")
        self.samples = [os.path.splitext(f)[0] for f in os.listdir(tensor_dir) if f.endswith(".npy")]
        self.samples.sort()

        # Split dataset
        random.seed(random_seed)
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * (1.0 - val_ratio))
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        # Load cropped lip tensor (T, 80, 80)
        tensor_path = os.path.join(self.output_dir, "lip_tensors", f"{sample_name}.npy")
        frames = np.load(tensor_path).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load transcript text
        txt_path = os.path.join(self.output_dir, "transcripts", f"{sample_name}.txt")
        with open(txt_path, "r") as f:
            sentence = f.read().strip()

        T = frames.shape[0]

        if self.window_size is not None and self.window_size < T:
            if self.window_mode == "subsample":
                # Sample window_size frames evenly distributed across the video
                indices = np.round(np.linspace(0, T - 1, self.window_size)).astype(int)
                frames = frames[indices]
                text = sentence
            elif self.window_mode == "slice":
                # Take a random contiguous window of window_size frames
                start_frame = random.randint(0, T - self.window_size)
                end_frame = start_frame + self.window_size
                frames = frames[start_frame:end_frame]

                # Adjust the transcript words that fall within this window
                # Each frame is 40ms, which is 1000 units in the alignment file (sample rate 25kHz)
                start_unit = start_frame * 1000
                end_unit = end_frame * 1000

                # Load align file to get exact word timings
                # If align file is not found, fallback to full sentence
                align_path = os.path.join(self.output_dir, "..", "grid_corpus", "s1", "align", f"{sample_name}.align")
                if os.path.exists(align_path):
                    words = []
                    with open(align_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 3:
                                word_start, word_end, word = int(parts[0]), int(parts[1]), parts[2]
                                if word not in ["sil", "sp"]:
                                    # If the word overlaps with our window, keep it
                                    if max(start_unit, word_start) < min(end_unit, word_end):
                                        words.append(word)
                    text = " ".join(words)
                else:
                    text = sentence
            else:
                text = sentence
        else:
            text = sentence

        # Add channel dimension: (T, 80, 80) -> (T, 1, 80, 80)
        frames = np.expand_dims(frames, axis=1)

        # Convert to tensors
        frames_tensor = torch.tensor(frames, dtype=torch.float32)
        target = torch.tensor(encode_text(text), dtype=torch.long)

        return frames_tensor, target

def collate_fn(batch):
    """
    Collate function to pad targets and inputs to form a batch.
    """
    inputs, targets = zip(*batch)
    
    # GRID inputs are already uniform length if window_size is fixed
    # But just in case, let's pad input sequence lengths
    input_lengths = [x.size(0) for x in inputs]
    max_input_len = max(input_lengths)
    
    padded_inputs = []
    for x in inputs:
        if x.size(0) < max_input_len:
            pad = torch.zeros(max_input_len - x.size(0), x.size(1), x.size(2), x.size(3))
            padded_inputs.append(torch.cat([x, pad], dim=0))
        else:
            padded_inputs.append(x)
            
    padded_inputs = torch.stack(padded_inputs, dim=0)  # (Batch, T, 1, 80, 80)
    
    # Pad target sequences (for CTC Loss)
    target_lengths = [len(y) for y in targets]
    max_target_len = max(target_lengths) if target_lengths else 1
    
    padded_targets = []
    for y in targets:
        if len(y) < max_target_len:
            pad = torch.zeros(max_target_len - len(y), dtype=torch.long)
            padded_targets.append(torch.cat([y, pad], dim=0))
        else:
            padded_targets.append(y)
            
    padded_targets = torch.stack(padded_targets, dim=0)  # (Batch, MaxTargetLen)
    
    return padded_inputs, padded_targets, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(target_lengths, dtype=torch.long)
