import torch
import torch.nn as nn


class DisModel(nn.Module):

    def __init__(self, hparams):

        super().__init__()

        in_channels = 3
        factor = 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=in_channels *
                      2, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels *
                      4, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(in_channels *
                           4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels * 4, out_channels=in_channels *
                      8, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(in_channels *
                           8),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels * 8, out_channels=in_channels *
                      16, kernel_size=4, padding=1, stride=4),
            nn.BatchNorm2d(in_channels *
                           16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels * 16,
                      out_channels=1, kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)
