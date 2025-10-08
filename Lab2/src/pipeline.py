from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    print("Lab 2 pipeline")
    train_transform = transforms.Compose([transforms.ToTensor()]) 

    train_set = OxfordIIITPet('./data/oxford_pet', download=True, 
    transform=train_transform)
    img, target = train_set[60]

    img_np = img.permute(1, 2, 0).numpy()  # Change from CHW to HWC format




if __name__ == '__main__':
    main()
    plt.show()