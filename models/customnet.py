from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network, experiment with number of layers
        # dim 3 because RGB
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2) #from 3 input channels to 64 output channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Add more layers...
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256*7*7,200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 112 x 112
        # print to see output shape after first convolutional layer
        x = self.pool(x)
        x = self.conv2(x).relu()
        x = self.pool(x)
        x = self.conv3(x).relu()
        x = self.pool(x)
        x = self.pool(x)
        print(x.shape) # 1:#batch(fixed, not needed) 2:#channels(256) 3,4:dim(7*7)
        x = x.view(x.size(0), -1) # Flatten before linear layer
        x = self.fc1(x)  # Pass through the fully connected layer

        return x