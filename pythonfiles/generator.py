import torch
import torch.nn as nn


class GenModel(nn.Module):

    def __init__(self, hparams):

        super(GenModel, self).__init__()

        self.hparams = hparams

        self.network = nn.Sequential(

            nn.ConvTranspose2d(
                self.hparams["z_shape"], self.hparams["final_layer_size"] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hparams["final_layer_size"] * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.hparams["final_layer_size"] * 8,
                               self.hparams["final_layer_size"] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams["final_layer_size"] * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.hparams["final_layer_size"] * 4,
                               self.hparams["final_layer_size"] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams["final_layer_size"] * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.hparams["final_layer_size"] * 2,
                               self.hparams["final_layer_size"], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams["final_layer_size"]),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                self.hparams["final_layer_size"], 3, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, x):

        return self.network(x)
