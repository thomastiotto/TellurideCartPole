import argparse, pickle, os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('../models')

from differentiable_plasticity_adaptation.models.MLP import MLP

dataset = 'CPS-17-02-2023-UpDown-Imitation-noise'


def main(args):
    with open(f'../exported/Train-{dataset}.pickle', 'rb') as file:  # loading training samples.
        training_samples = pickle.load(file)

    with open(f'../exported/Test-{dataset}.pickle', 'rb') as file:  # loading training samples.
        test_samples = pickle.load(file)

    mlp = MLP(args)  # instantiate ANN.

    mlp.export_parameters(phase='before')

    mlp.train(training_samples)  # train model.

    mlp.export_parameters(phase='after')

    plt.plot(np.arange(0, len(mlp.losses), 1), mlp.losses, label='MLP')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend(loc='best', framealpha=0)
    plt.ylim(0, 0.6)
    plt.axhline(y=0.025, color='grey', linestyle='--')
    plt.show()
    plt.close()

    losses = mlp.predict(test_samples)  # test model.

    print(f'average loss: {np.mean(losses)}')

    plt.scatter(np.arange(0, len(losses), 1), losses, label='MLP', marker='*', alpha=0.5)
    plt.title(f'average loss: {np.mean(losses)}')
    plt.xlabel('testing')
    plt.ylabel('MSE')
    plt.legend(loc='best', framealpha=0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP Differential Plasticity')

    parser.add_argument('--input_size', type=int, default=8, help='Input size')
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size')
    parser.add_argument('--output_size', type=int, default=1, help='Output size')
    parser.add_argument('--depth', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--show_loss', type=int, default=1, help='Shows loss over training')

    args = parser.parse_args()

    main(args)
