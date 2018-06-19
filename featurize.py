from models.auto_enc import AE
import numpy as np
import torch

model = AE()
state = torch.load('checkpoints/ae_10.pth.tar')
model.load_state_dict(state['state_dict'])
games = np.load('data/bitboards.npy')

# got lucky, 17 divides number of samples evenly
batched_games = np.split(games, 17) 

def featurize(game):
    recon, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()

feat_games = [featurize(batch) for batch in batched_games]
featurized = np.vstack(feat_games)

np.save('./data/features.npy', featurized)
