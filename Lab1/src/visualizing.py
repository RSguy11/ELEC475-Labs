import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer
import numpy as np
import torch

def main():

    userInt = int(input("Enter an integer between 0 and 60000: "))
    print("You entered: ", userInt)
    
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    
    train_set = MNIST('./data/mnist', train=True, download=True, 
    transform=train_transform)



    img = train_set.data[userInt]
    image_size = 28 * 28
    model = autoencoderMLP4Layer(image_size)

    model.load_state_dict(torch.load(
        'MLP.8.pth', map_location=torch.device('cpu')))
    
    model.eval()

    img_normalized = img.type(torch.float32) / 255.0
    noise_ratio = 0.5

    noise = (torch.rand(28, 28)-0.5)*2 * noise_ratio
    noise_img = noise + img_normalized
    noise_img = torch.clamp(noise_img, 0., 1.)    

    with torch.no_grad():
        output = model(noise_img.flatten())
    
    output = output.view(28, 28)

    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    f.add_subplot(1,3,2)
    plt.imshow(noise_img, cmap='gray')
    f.add_subplot(1,3,3)
    plt.imshow(output, cmap='gray')
    plt.show()


def intermolationMod():
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    
    train_set = MNIST('./data/mnist', train=True, download=True, 
    transform=train_transform)
    img1 = train_set.data[50].type(torch.float32) / 255.0
    img2 = train_set.data[100].type(torch.float32) / 255.0

    image_size = 28 * 28
    model = autoencoderMLP4Layer(image_size)
    model.load_state_dict(torch.load(
        'MLP.8.pth', map_location=torch.device('cpu')))
    
    model.eval()

    bottle1 = model.encode(img1.flatten())
    bottle2 = model.encode(img2.flatten())

    f = plt.figure()
    f.add_subplot(1,10,1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    for i, alpha in enumerate(np.linspace(0, 1, 8)):
        inter = (1 - alpha) * bottle1 + alpha * bottle2
        decoded = model.decode(inter).view(28, 28).detach().numpy()
        f.add_subplot(1,10,i+2)
        plt.imshow(decoded, cmap='gray')
        plt.axis('off')

    f.add_subplot(1,10,10)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    #main()
    intermolationMod()

