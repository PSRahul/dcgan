import torch
import torch.nn as nn


class GenModel(nn.Module):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        self.network = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=self.hparams["z_shape"], out_channels=self.hparams["gen_final_layer_size"] * 4, kernel_size=3),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 4),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 4, out_channels=self.hparams["gen_final_layer_size"] * 8, kernel_size=3),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 8),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 8, out_channels=self.hparams["gen_final_layer_size"] * 12, kernel_size=3),  # stride=2, padding=1),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 12),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 12, out_channels=self.hparams["gen_final_layer_size"] * 16, kernel_size=3),  # stride=2, padding=1),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 16),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 16, out_channels=self.hparams["gen_final_layer_size"] * 24, kernel_size=3),  # stride=2, padding=1),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 24),
            nn.PReLU(),
            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 24, out_channels=self.hparams["gen_final_layer_size"] * 32, kernel_size=3),  # stride=2),# padding=1),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 32),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 32, out_channels=self.hparams["gen_final_layer_size"] * 48, kernel_size=4),  # stride=2, padding=1),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 48),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 48, out_channels=self.hparams["gen_final_layer_size"] * 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hparams["gen_final_layer_size"] * 64),
            nn.PReLU(),

            nn.ConvTranspose2d(
                in_channels=self.hparams["gen_final_layer_size"] * 64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()



        )

    def forward(self, x):

        return self.network(x)
