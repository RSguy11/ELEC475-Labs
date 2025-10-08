import torch 
import torch.nn as nn
import torch.nn.functional as F

class SnoutNet(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, input_size=227):
        super(SnoutNet, self).__init__()

        #Defining the Convolution Layers, Using the number of feature layers as the bounds
        #Each Layer learns a different layer of extraction with each convolution layer taking in details refined from the previous layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)

        #Defining the Fully Connected Layers
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x shape at this pont (batch_size, 256,4,4)

        x = x.view(x.size(0), -1) #flattening the tensor # → [batch, 4096]
        x = F.relu(self.fc1(x)) # → [batch, 1024]
        x = F.relu(self.fc2(x)) # → [batch, 1024]
        x = self.fc3(x) # → [batch, 2]

