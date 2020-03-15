import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        # Spectros come in at 40 x 256
        self.conv1 = nn.Conv2d(1, 6, 3) # 6x38x254
        self.conv2 = nn.Conv2d(6, 16, 3) # 16x36x252 -> Maxpool.2= 16x18x126
        self.conv3 = nn.Conv2d(16, 1, 5) # 16x36x252 -> Maxpool.2= 16x18x126
        self.fc1 = nn.Linear(14*122, 50) # 120
        # self.fc2 = nn.Linear(120, 50) # 50
        self.fc3 = nn.Linear(50, 256) # 256 final out

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x, 0.1)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        # output = F.softmax(x, dim=1)
        output = F.log_softmax(x, dim=1)
        return output
