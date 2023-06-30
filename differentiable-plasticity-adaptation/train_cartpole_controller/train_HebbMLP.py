import argparse, pickle, os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../models')

from models.Hebb_MLP import HebbMLP

dataset='27s'

def main(args):

	with open(f'../datasets/Train-{dataset}-noise.pickle', 'rb') as file:		# loading training samples.
		training_samples = pickle.load(file)

	with open(f'../datasets/Test-{dataset}-noise.pickle', 'rb') as file:		# loading training samples.
		test_samples = pickle.load(file)

	args.input_size = len(training_samples['X'][0])
	args.hidden_size = int(2*len(training_samples['X'][0]))

	hebbmlp = HebbMLP(args)											# instantiate ANN.

	hebbmlp.train(training_samples)									# train model.

	plt.plot(np.arange(0, len(hebbmlp.losses), 1), hebbmlp.losses, label = 'Hebb-MLP')
	plt.xlabel('epochs')
	plt.ylabel('MSE')
	plt.legend(loc = 'best', framealpha = 0)
	plt.xlim(0, len(hebbmlp.losses))
	plt.ylim(0, 0.6)
	plt.axhline(y = 0.025, color = 'grey', linestyle = '--')
	plt.show()
	plt.close()

	losses = hebbmlp.predict(test_samples)							# test model.
	print(f'average loss: {np.mean(losses)}')

	plt.scatter(np.arange(0, len(losses), 1), losses, label = 'Hebb-MLP', marker = '*', alpha = 0.5)
	plt.title(f'average loss: {np.mean(losses)}')
	plt.xlabel('testing')
	plt.ylabel('MSE')
	plt.legend(loc = 'best', framealpha = 0)
	plt.show()
	plt.close()

	hebbmlp.export_parameters('cartpole-trained')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'MLP Differential Plasticity')

	parser.add_argument('--input_size', type = int, default = 2, help = 'Input size')
	parser.add_argument('--hidden_size', type = int, default = 10, help = 'Hidden size')
	parser.add_argument('--output_size', type = int, default = 1, help = 'Output size')
	parser.add_argument('--depth', type = int, default = 1, help = 'Number of hidden layers')
	parser.add_argument('--lr', type = float, default = 2e-3, help = 'Learning rate')
	parser.add_argument('--epochs', type = int, default = 100, help = 'Number of epochs')
	parser.add_argument('--show_loss', type = int, default = 1, help = 'Shows loss over training')

	args = parser.parse_args()

	main(args)
