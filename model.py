import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1), # (32, 256, 256)
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2), # (32, 128, 128)

            nn.Conv2d(32, 64, 3, stride=1, padding=1), # (64, 128, 128)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # (64, 64, 64)

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # (128, 64, 64)
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2) # (128, 32, 32)


        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # (64, 64, 64)
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # (32, 128, 128)
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), # (3, 256, 256)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
