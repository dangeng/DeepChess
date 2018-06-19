from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#writer = SummaryWriter()

games = np.load('../data/bitboards.npy')
wins = np.load('../data/labels.npy')

#np.random.shuffle(games)
train_games = games[:int(len(games)*.8)]
train_wins = wins[:int(len(games)*.8)]
test_games = games[int(len(games)*.8):]
test_wins = wins[int(len(games)*.8):]

train_games_wins = train_games[train_wins == 1]
train_games_losses = train_games[train_wins == -1]

test_games_wins = test_games[test_wins == 1]
test_games_losses = test_games[test_wins == -1]

raise Exception('stop here')

class TrainSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        rand_win = train_games_wins[np.random.randint(0, train_games_wins.shape[0])]
        rand_loss = train_games_losses[np.random.randint(0, train_games_losses.shape[0])]

        rand_win = torch.from_numpy(rand_win).type(torch.FloatTensor)
        rand_loss = torch.from_numpy(rand_loss).type(torch.FloatTensor)

        order = np.random.randint(0,2)
        if order == 0:
            return ((rand_win, rand_loss), 0)
        else:
            return ((rand_loss, rand_win), 1)

    def __len__(self):
        return self.length

class TestSet(Dataset):
    def __init__(length):
        self.length = length

    def __getitem__(self, index):
        rand_win = test_games_wins[np.random.randint(0, test_games_wins.shape[0])]
        rand_loss = test_games_losses[np.random.randint(0, test_games_losses.shape[0])]

        rand_win = torch.from_numpy(rand_win).type(torch.FloatTensor)
        rand_loss = torch.from_numpy(rand_loss).type(torch.FloatTensor)

        order = np.random.randint(0,2)
        if order == 0:
            return ((rand_win, rand_loss), 0)
        else:
            return ((rand_loss, rand_win), 1)

    def __len__(self):
        return self.length

train_loader = torch.utils.data.DataLoader(TrainSet(1000000),batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(10000),batch_size=128, shuffle=True)

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, pos1, pos2):
        x = torch.cat(pos1, pos2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.softmax(x)


model = Siamese().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 773), size_average=False)

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        pos1, pos2 = data
        pos1, pos2 = pos1.to(device), pos2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(recon_batch, label)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            writer.add_scalar('data/train_loss', loss.item() / len(data), epoch*len(train_loader) + batch_idx)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, enc = model(data)
            test_loss += loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('data/test_loss', test_loss, epoch)

def save(epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    torch.save(state, '../checkpoints/ae_{}.pth.tar'.format(epoch + 1))

def adjust_learning_rate(optimizer):
    ''' Divide learning rate by 10 '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

def recon(game):
    recon, _ = model(torch.from_numpy(game).type(torch.FloatTensor))
    recon = (recon.detach().numpy() > .5).astype(int)
    return recon

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    save(epoch)

