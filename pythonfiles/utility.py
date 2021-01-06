import torch


def init_weights(m):
    if type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
