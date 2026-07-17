import torch
import torch.nn as nn
import torchsparse.nn as spnn
from torchsparse.backbones.modules.blocks import SparseResBlock

class SparseClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            spnn.Conv3d(8, 32, kernel_size=3, stride=1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )
        self.layer1 = nn.Sequential(
            SparseResBlock(32, 32, 3),
            SparseResBlock(32, 32, 3)
        )
        self.layer2 = nn.Sequential(
            spnn.Conv3d(32, 64, kernel_size=3, stride=2), # Downsample
            spnn.BatchNorm(64),
            spnn.ReLU(True),
            SparseResBlock(64, 64, 3)
        )
        self.layer3 = nn.Sequential(
            spnn.Conv3d(64, 128, kernel_size=3, stride=2), # Downsample
            spnn.BatchNorm(128),
            spnn.ReLU(True),
            SparseResBlock(128, 128, 3)
        )
        self.pool = spnn.GlobalMaxPool()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.classifier(out)
