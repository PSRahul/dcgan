
from pythonfiles.generator import GenModel

from torchsummary import summary
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision  
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--d1', required=True)
parser.add_argument('--d2', required=True)
args = parser.parse_args()

hparams = {
    'z_shape': 128,
    'gen_final_layer_size': 3,
    'image_input_shape': 64,
    'batch_size': 128,
    'epochs': 5,
    'dropout_1':float(args.d1),
    'dropout_2':float(args.d2),
    
}
savestring="dp1"+str(int(hparams['dropout_1']*10))+"dp2"+str(int(hparams['dropout_2']*10))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using the  Device - ", device)


gen_model = GenModel(hparams)
gen_model = gen_model.to(device)
gen_model.load_state_dict(torch.load("models/gen"+str(4)+savestring))
gen_model.eval()

seed=2563
torch.manual_seed(seed)
vis_noise = torch.randn(1, hparams['z_shape'], 1, 1, device=device)
with torch.no_grad():
        img_out_full=gen_model(vis_noise)
        torchvision.utils.save_image(img_out_full,fp="samples/"+str(seed)+savestring+".png",normalize=True,nrow=1)





