import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, dropout_rate=0.1, num_classes=7, num_input_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1  = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2  = nn.Linear(64, num_classes)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = self.gap(x).view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
