import argparse, pickle, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../models')

from Hebb_MLP import HebbMLP
from MLP import MLP

def main(args):

	# loading training samples.

	with open('../datasets/bin_classification_data.pickle', 'rb') as file:
		training_samples = pickle.load(file)

	_hebbmlploss = []
	_mlploss = []

	for i in range(10):

		# training Hebb-MLP.

		hebbmlp = HebbMLP(args)											# instantiate Hebb-ANN.
		hebbmlp.train(training_samples)									# train model.

		_hebbmlploss.append(hebbmlp.losses)

		# training MLP.

		mlp = MLP(args)													# instantiate ANN.
		mlp.train(training_samples)										# train model.

		_mlploss.append(mlp.losses)

	# compare training performance.

	plt.plot(np.arange(0, len(hebbmlp.losses), 1), np.mean(_hebbmlploss, axis = 0), label = 'Hebb-MLP')
	plt.fill_between(
		np.arange(0, len(hebbmlp.losses), 1), 
		np.mean(_hebbmlploss, axis = 0) - np.std(_hebbmlploss, axis = 0), 
		np.mean(_hebbmlploss, axis = 0) + np.std(_hebbmlploss, axis = 0), 
		alpha = 0.5)

	plt.plot(np.arange(0, len(mlp.losses), 1), np.mean(_mlploss, axis = 0), color = 'r', label = 'MLP')
	plt.fill_between(
		np.arange(0, len(mlp.losses), 1), 
		np.mean(_mlploss, axis = 0) - np.std(_mlploss, axis = 0), 
		np.mean(_mlploss, axis = 0) + np.std(_mlploss, axis = 0), 
		alpha = 0.5,
		color = 'r')

	plt.xlabel('epochs')
	plt.ylabel('MSE')

	plt.xlim(0, len(hebbmlp.losses))
	plt.ylim(0, 0.6)

	plt.legend(loc = 'best', framealpha = 0.0)

	plt.show()
	plt.close()

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
