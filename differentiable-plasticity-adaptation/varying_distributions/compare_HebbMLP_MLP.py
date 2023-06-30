import argparse, pickle, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../models')

from Hebb_MLP import HebbMLP
from MLP import MLP

def main(args):

	# loading training samples.

	with open('../datasets/normal_data_train.pickle', 'rb') as file:
		training_samples = pickle.load(file)

	# training Hebb-MLP.

	hebbmlp = HebbMLP(args)											# instantiate Hebb-ANN.
	hebbmlp.train(training_samples)									# train model.

	# training MLP.

	mlp = MLP(args)													# instantiate ANN.
	mlp.train(training_samples)										# train model.

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'MLP Differential Plasticity')

	parser.add_argument('--input_size', type = int, default = 5, help = 'Input size')
	parser.add_argument('--hidden_size', type = int, default = 50, help = 'Hidden size')
	parser.add_argument('--output_size', type = int, default = 1, help = 'Output size')
	parser.add_argument('--depth', type = int, default = 2, help = 'Number of hidden layers')
	parser.add_argument('--lr', type = float, default = 2e-2, help = 'Learning rate')
	parser.add_argument('--epochs', type = int, default = 1000, help = 'Number of epochs')
	parser.add_argument('--show_loss', type = int, default = 1, help = 'Shows loss over training')

	args = parser.parse_args()

	main(args)
