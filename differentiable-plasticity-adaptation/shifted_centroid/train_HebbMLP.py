import argparse, pickle, os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../models')

from Hebb_MLP import HebbMLP

def main(args):

	with open('../datasets/shifted_distribution_centroid-train.pickle', 'rb') as file:		# loading training samples.
		training_samples = pickle.load(file)

	with open('../datasets/shifted_distribution_centroid-test.pickle', 'rb') as file:		# loading training samples.
		test_samples = pickle.load(file)

	hebbmlp = HebbMLP(args)											# instantiate ANN.

	hebbmlp.train(training_samples)									# train model.

	# plt.plot(np.arange(0, len(hebbmlp.losses), 1), hebbmlp.losses)
	# plt.xlabel('epochs')
	# plt.ylabel('MSE')
	# plt.show()
	# plt.close()

	losses = hebbmlp.predict(test_samples)							# test model.

	print(np.mean(losses))

	plt.plot(np.arange(0, len(losses), 1), losses)
	plt.xlabel('testing')
	plt.ylabel('MSE')
	plt.show()
	plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'MLP Differential Plasticity')

	parser.add_argument('--input_size', type = int, default = 2, help = 'Input size')
	parser.add_argument('--hidden_size', type = int, default = 50, help = 'Hidden size')
	parser.add_argument('--output_size', type = int, default = 1, help = 'Output size')
	parser.add_argument('--depth', type = int, default = 2, help = 'Number of hidden layers')
	parser.add_argument('--lr', type = float, default = 2e-3, help = 'Learning rate')
	parser.add_argument('--epochs', type = int, default = 500, help = 'Number of epochs')
	parser.add_argument('--show_loss', type = int, default = 1, help = 'Shows loss over training')

	args = parser.parse_args()

	main(args)
