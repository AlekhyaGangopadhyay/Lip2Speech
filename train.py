import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LipReadingDataset, collate_fn, decode_text
from model import LipNet

def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_metrics(preds, targets):
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    
    for p, t in zip(preds, targets):
        total_char_dist += edit_distance(p, t)
        total_chars += len(t)
        
        p_words = p.split()
        t_words = t.split()
        total_word_dist += edit_distance(p_words, t_words)
        total_words += len(t_words)
        
    cer = total_char_dist / max(1, total_chars)
    wer = total_word_dist / max(1, total_words)
    return cer, wer

def greedy_decode(outputs):
    # outputs shape: (Batch, T, VocabSize)
    decoded_sentences = []
    arg_maxes = torch.argmax(outputs, dim=-1)  # (Batch, T)
    for indices in arg_maxes:
        collapsed = []
        prev = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev:
                if idx != 0:  # skip blank token
                    collapsed.append(idx)
                prev = idx
        decoded_sentences.append(decode_text(collapsed))
    return decoded_sentences

def train():
    # Configurations
    epochs = 15
    batch_size = 8
    learning_rate = 1e-4
    window_size = 29
    window_mode = "subsample"  # "subsample" or "slice"
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = LipReadingDataset(split="train", val_ratio=0.2, window_size=window_size, window_mode=window_mode)
    val_dataset = LipReadingDataset(split="val", val_ratio=0.2, window_size=window_size, window_mode=window_mode)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize model
    # Vocab size: 28 (0: blank, 1: space, 2..27: a-z)
    model = LipNet(vocab_size=28).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # CTCLoss expects inputs of shape (T, Batch, VocabSize) if batch_first=False
    # zero_infinity=True prevents loss from exploding if the target sequence is longer than the input sequence
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_loss = float("inf")
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            inputs = inputs.to(device)                  # (Batch, T, 1, 80, 80)
            targets = targets.to(device)                # (Batch, MaxTargetLen)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)                     # (Batch, T, VocabSize)
            
            # Transpose to (T, Batch, VocabSize) for CTCLoss
            outputs_ctc = outputs.transpose(0, 1)       # (T, Batch, VocabSize)
            
            loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, input_lengths, target_lengths in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                outputs_ctc = outputs.transpose(0, 1)
                
                loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)
                val_loss += loss.item() * inputs.size(0)
                
                # Decode predictions
                preds = greedy_decode(outputs)
                all_preds.extend(preds)
                
                # Decode ground truths
                for target_seq, length in zip(targets, target_lengths):
                    tgt_indices = target_seq[:length].tolist()
                    all_targets.append(decode_text(tgt_indices))
                    
        epoch_val_loss = val_loss / len(val_dataset)
        cer, wer = compute_metrics(all_preds, all_targets)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val CER: {cer:.4f} | Val WER: {wer:.4f} | Time: {elapsed_time:.1f}s")
        
        # Show a few sample predictions
        if len(all_preds) > 0:
            print("  Samples:")
            for i in range(min(3, len(all_preds))):
                print(f"    Target: {all_targets[i]}")
                print(f"    Pred:   {all_preds[i]}")
                print("    ---")
                
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_lipnet.pth")
            print(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")

    print("Training finished!")

if __name__ == "__main__":
    train()
