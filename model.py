import torch
import torch.nn as nn

class ImageSequentialClassifier(nn.Module):

    def __init__(self) -> None:
        super(ImageSequentialClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(7488, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.digit_length = nn.Sequential(nn.Linear(1024, 7))
        self.digit1 = nn.Sequential(nn.Linear(1024, 11))
        self.digit2 = nn.Sequential(nn.Linear(1024, 11))
        self.digit3 = nn.Sequential(nn.Linear(1024, 11))
        self.digit4 = nn.Sequential(nn.Linear(1024, 11))
        self.digit5 = nn.Sequential(nn.Linear(1024, 11))

        self.bbox1 = nn.Sequential(nn.Linear(1024, 4))
        self.bbox2 = nn.Sequential(nn.Linear(1024, 4))
        self.bbox3 = nn.Sequential(nn.Linear(1024, 4))
        self.bbox4 = nn.Sequential(nn.Linear(1024, 4))
        self.bbox5 = nn.Sequential(nn.Linear(1024, 4))

    def forward(self, x):
        # x : (N, C, H, W)
        x = self.encoder(x) # (N, 1024)
        length_logits = self.digit_length(x)
        digit1_logits = self.digit1(x)
        digit2_logits = self.digit2(x)
        digit3_logits = self.digit3(x)
        digit4_logits = self.digit4(x)
        digit5_logits = self.digit5(x)

        bbox1_logits = self.bbox1(x)
        bbox2_logits = self.bbox2(x)
        bbox3_logits = self.bbox3(x)
        bbox4_logits = self.bbox4(x)
        bbox5_logits = self.bbox5(x)

        return length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits,\
        bbox1_logits, bbox2_logits, bbox3_logits, bbox4_logits, bbox5_logits
    
    