import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

def main():

    userInt = int(input("Enter an integer between 0 and 60000: "))
    print("You entered: ", userInt)
    
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    
    train_set = MNIST('./data/mnist', train=True, download=True, 
    transform=train_transform)

    plt.imshow(train_set.data[userInt], cmap='gray') 
    plt.show() 



if __name__ == '__main__':
    main()
