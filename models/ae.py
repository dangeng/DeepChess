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
#games = games - np.mean(games, axis=0)

np.random.shuffle(games)
train_games = games[:int(len(games)*.8)]
#train_games = games[:10000]
test_games = games[int(len(games)*.8):]
#test_games = games[-10:]

class TrainSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return (torch.from_numpy(train_games[index]).type(torch.FloatTensor), 1)

    def __len__(self):
        return train_games.shape[0]

class TestSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return (torch.from_numpy(test_games[index]).type(torch.FloatTensor), 1)

    def __len__(self):
        return test_games.shape[0]

train_loader = torch.utils.data.DataLoader(TrainSet(),batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(),batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fce1 = nn.Linear(773, 600)
        self.fce2 = nn.Linear(600, 400)
        self.fce3 = nn.Linear(400, 200)
        self.fce4 = nn.Linear(200, 100)

        self.fcd1 = nn.Linear(100, 200)
        self.fcd2 = nn.Linear(200, 400)
        self.fcd3 = nn.Linear(400, 600)
        self.fcd4 = nn.Linear(600, 773)

    def encode(self, x):
        x = F.leaky_relu(self.fce1(x))
        x = F.leaky_relu(self.fce2(x))
        x = F.leaky_relu(self.fce3(x))
        x = F.leaky_relu(self.fce4(x))
        return x

    def decode(self, z):
        z = F.leaky_relu(self.fcd1(z))
        z = F.leaky_relu(self.fcd2(z))
        z = F.leaky_relu(self.fcd3(z))
        z = F.sigmoid(self.fcd4(z))
        return z

    def forward(self, x):
        enc = self.encode(x.view(-1, 773))
        return self.decode(enc), enc


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

state = torch.load('../checkpoints/ae_6.pth.tar')
model.load_state_dict(state['state_dict'])
raise Exception('stop')

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 773), size_average=False)

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, enc = model(data)
        loss = loss_function(recon_batch, data)
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
    torch.save(state, '../checkpoints/ae_{}.pth.tar'.format(epoch))

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

