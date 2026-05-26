import torch
import torch.nn as nn
import torch.nn.functional as F

class LipNet(nn.Module):
    def __init__(self, vocab_size=28):
        super(LipNet, self).__init__()
        
        # 3D Convolutional layers to capture spatiotemporal features
        # Input shape: (Batch, 1, T, 80, 80)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Spatial dimensions at the end of Conv3D:
        # Height: 80 -> (/2 pool1) -> 40 -> (/2 pool2) -> 20 -> (/2 pool3) -> 10. Wait, stride=2 in conv1 makes it 40, pool1 makes it 20.
        # Let's trace height/width:
        # conv1 output: (80 + 2*3 - 7)/2 + 1 = 80/2 = 40.
        # pool1 output: 20
        # conv2 output: 20
        # pool2 output: 10
        # conv3 output: 10
        # pool3 output: 5.
        # So final spatial size is 5x5.
        # Flattened spatial size per frame: 96 channels * 5 * 5 = 2400
        
        self.gru = nn.GRU(
            input_size=2400,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        
        self.fc = nn.Linear(512, vocab_size)  # 256 * 2 (bidirectional) -> vocab_size
        
    def forward(self, x):
        # x shape: (Batch, T, 1, 80, 80)
        # Permute to (Batch, 1, T, 80, 80) for Conv3D
        x = x.transpose(1, 2)
        
        # 3D CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Output shape: (Batch, Channels=96, T, H=5, W=5)
        # Permute and reshape to (Batch, T, Channels * H * W)
        b, c, t, h, w = x.size()
        x = x.transpose(1, 2).contiguous()  # (Batch, T, Channels, H, W)
        x = x.view(b, t, c * h * w)         # (Batch, T, 2400)
        
        # Bi-GRU
        x, _ = self.gru(x)                  # (Batch, T, 512)
        
        # Output Log-probabilities
        x = self.fc(x)                      # (Batch, T, VocabSize)
        
        # Apply LogSoftmax for CTC Loss
        return F.log_softmax(x, dim=-1)
