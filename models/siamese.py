from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.fc1 = nn.Linear(200, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 2)
        self.bn4 = nn.BatchNorm1d(2)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.bn4(self.fc4(x))
        return F.sigmoid(x)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 2), size_average=False)
    return BCE

