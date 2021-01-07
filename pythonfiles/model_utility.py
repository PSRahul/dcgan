import torch


def init_weights(m):
    classname = m.__class__.__name__

    if type(m) == torch.nn.Conv2d:
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.normal_(m.weight, mean=0, std=0.02)

    if type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.normal_(m.weight, mean=0, std=0.02)

    if type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight, mean=1, std=0.02)
        m.bias.data.zero_()
