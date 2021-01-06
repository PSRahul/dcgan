import torch

def init_weights(m):
    classname = m.__class__.__name__
       
    if type(m)== torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

    if type(m)== torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)

    if type(m)== torch.nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight)
        m.bias.data.zero_()


