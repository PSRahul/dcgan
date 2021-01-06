from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms



def data_setup(batch_size=4, image_size=64):
    image_path = os.path.join(os.getcwd(), "data/subsetdata")
    print(image_path)
    image_data = ImageFolder(root=image_path,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor()]))

    data_loader = DataLoader(image_data,
                             batch_size=batch_size,
                             num_workers=6)

    return data_loader
