import argparse, pickle, os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from tqdm.auto import tqdm

import model_def

from differentiable_plasticity_adaptation.models.MLP import MLP

dataset = 'CPS-17-02-2023-UpDown-Imitation-noise'

parser = argparse.ArgumentParser(description='MLP Differential Plasticity')

parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size')
parser.add_argument('--depth', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--show_loss', type=int, default=1, help='Shows loss over training')

args = parser.parse_args()

with open(f'../datasets/Train-{dataset}.pickle', 'rb') as file:  # loading training samples.
    training_samples = pickle.load(file)

with open(f'../datasets/Test-{dataset}.pickle', 'rb') as file:  # loading training samples.
    test_samples = pickle.load(file)

input_size = len(training_samples[0]['X'][0])

model = model_def.define_rec_model_struct(args.hidden_size, input_size, args.depth)

summary(model, input_size=(32, input_size))

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

e_loss = []
for e in range(args.epochs):
    for exp in tqdm(training_samples, leave=False, position=0):
        exp_loss = []
        for i in range(0, len(exp['X']), 32):
            Xbatch = torch.tensor(exp['X'][i:i + 32], dtype=torch.float32)
            Ybatch = torch.tensor(exp['Y'][i:i + 32], dtype=torch.float32)
            Ypred = model(Xbatch)
            loss = loss_fn(Ypred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp_loss.append(loss.item())
    e_loss.append(np.mean(exp_loss))
    print(f'epoch {e} loss: {e_loss[-1]}')

if args.show_loss:
    plt.plot(e_loss)
    plt.show()
    plt.savefig(f'../exported/MLP_loss-{dataset}-after.png')

torch.save(model.state_dict(), f'../exported/MLP_parameters-{dataset}-after')
print(f'Model saved to ../exported/MLP_parameters-{dataset}-after')

with torch.no_grad():
    y_pred = model(torch.Tensor(test_samples[0]['X']))
print('Validation loss', loss_fn(y_pred, torch.Tensor(test_samples[0]['Y'])))
