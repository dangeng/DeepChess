from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from tensorboardX import SummaryWriter
from models.siamese import Siamese

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=.01, metavar='N',
                    help='learning rate (default: .01)')
parser.add_argument('--decay', type=int, default=.99, metavar='N',
                    help='decay rate of learning rate (default: .99)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

lr = args.lr
decay = args.decay
batch_size = args.batch_size

writer = SummaryWriter(comment='lr: {} | decay: {} | batch size: {}'.format(lr, decay, batch_size))

print('Loading data...')
games = np.load('./data/features.npy')
wins = np.load('./data/labels.npy')

p = np.random.permutation(len(wins))
games = games[p]
wins = wins[p]

train_games = games[:int(len(games)*.8)]
train_wins = wins[:int(len(games)*.8)]
test_games = games[int(len(games)*.8):]
test_wins = wins[int(len(games)*.8):]

train_games_wins = train_games[train_wins == 1]
train_games_losses = train_games[train_wins == -1]

test_games_wins = test_games[test_wins == 1]
test_games_losses = test_games[test_wins == -1]

class TrainSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        rand_win = train_games_wins[
            np.random.randint(0, train_games_wins.shape[0])]
        rand_loss = train_games_losses[
            np.random.randint(0, train_games_losses.shape[0])]

        #rand_win = train_games_wins[0]
        #rand_loss = train_games_losses[1234]

        order = np.random.randint(0,2)
        if order == 0:
            stacked = np.hstack((rand_win, rand_loss))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return (stacked, label)
        else:
            stacked = np.hstack((rand_loss, rand_win))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return (stacked, label)

    def __len__(self):
        return self.length

class TestSet(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        rand_win = test_games_wins[np.random.randint(0, test_games_wins.shape[0])]
        rand_loss = test_games_losses[np.random.randint(0, test_games_losses.shape[0])]

        order = np.random.randint(0,2)
        if order == 0:
            stacked = np.hstack((rand_win, rand_loss))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return (stacked, label)
        else:
            stacked = np.hstack((rand_loss, rand_win))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return (stacked, label)

    def __len__(self):
        return self.length

train_loader = torch.utils.data.DataLoader(TrainSet(1000000),batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(100000),batch_size=batch_size, shuffle=True)


print('Buidling model...')
model = Siamese().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

e = enumerate(train_loader)
b, (data, label) = next(e)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(pred, label):
    BCE = F.binary_cross_entropy(pred, label, size_average=False)
    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = loss_function(pred, label)

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

def get_acc():
    e = enumerate(test_loader)
    correct = 0
    for batch_idx, (data, label) in e:
        pred = model(data.to(device))
        correct += np.sum((pred > .5).cpu().detach().numpy() * label.numpy())
    return correct / float(test_loader.dataset.length)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            test_loss += loss_function(pred, label).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('data/test_loss', test_loss, epoch)

def save(epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    save_dir = 'checkpoints/siamese/lr_{}_decay_{}'.format(int(lr*1000), int(decay*100))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'ae_{}.pth.tar'.format(epoch)))

start_epoch = 1
resume = True
if resume:
    state = torch.load('./checkpoints/best_siamese.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']

print('Begin train...')
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    test(epoch)
    save(epoch)

    # Adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay

