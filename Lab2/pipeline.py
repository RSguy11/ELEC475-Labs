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

    img = train_set.data[50]
    image_size = 28 * 28

    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(img, cmap='gray')



if __name__ == '__main__':
    main()
    plt.show()