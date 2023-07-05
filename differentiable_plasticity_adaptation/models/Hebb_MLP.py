import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
from tqdm.auto import tqdm

class HebbMLP:
	def __init__(self, args):

		self.__input_size = args.input_size
		self.__hidden_size = args.hidden_size
		self.__output_size = args.output_size
		self.__depth = args.depth					# number of hidden layers.

		self.__lr = args.lr							# learning rate.
		self.__epochs = args.epochs					# number of epochs.

		self.s1 = .01								# scaling factor for fixed weight and alpha.
		self.alpha_min = -1.0
		self.alpha_max = 1.0

		self.show_loss = args.show_loss

		self.__w = nn.ParameterList()				# fixed weights.
		self.__Hebb = []							# hebbian terms.
		self.__alpha = nn.ParameterList()			# hebbian term parameters.
		self.__y = []								# units' activations.

		self.__init_ANN()

		self.losses = []

	def __init_ANN(self):

		self.__init_parameters()

		self.optimizer = torch.optim.Adam(list(self.__w) + list(self.__alpha), lr = self.__lr)

	def __init_parameters(self):

		# w/alpha/Hebb between input and 1st hidden layer.

		self.__w.append(nn.Parameter(
			self.s1 * torch.randn(
				(self.__input_size, self.__hidden_size), requires_grad = True)))

		self.__Hebb.append(
			torch.zeros((self.__input_size, self.__hidden_size), requires_grad = False))

		self.__alpha.append(nn.Parameter(
			self.s1 * torch.randn(
				(self.__input_size, self.__hidden_size), requires_grad = True)))

		# w/alpha/Hebb between hidden layers.

		for l in range(self.__depth-1):

			self.__w.append(nn.Parameter(
				self.s1 * torch.randn(
					(self.__hidden_size, self.__hidden_size), requires_grad = True)))

			self.__Hebb.append(
				torch.zeros((self.__hidden_size, self.__hidden_size), requires_grad = False))

			self.__alpha.append(nn.Parameter(
				self.s1 * torch.randn(
					(self.__hidden_size, self.__hidden_size), requires_grad = True)))

		# w/alpha/Hebb between last hidden layer and output.

		self.__w.append(nn.Parameter(
			self.s1 * torch.randn(
					(self.__hidden_size, self.__output_size), requires_grad = True)))

		self.__Hebb.append(
				torch.zeros((self.__hidden_size, self.__output_size), requires_grad = False))

		self.__alpha.append(nn.Parameter(
			self.s1 * torch.randn(
				(self.__hidden_size, self.__output_size), requires_grad = True)))

	def __reset_Hebb(self):

		self.__Hebb = []

		self.__Hebb.append(
			torch.zeros((self.__input_size, self.__hidden_size), requires_grad = False))

		for l in range(self.__depth-1):

			self.__Hebb.append(
				torch.zeros((self.__hidden_size, self.__hidden_size), requires_grad = False))

		self.__Hebb.append(
				torch.zeros((self.__hidden_size, self.__output_size), requires_grad = False))

	def __reset_units(self):

		self.__y = []

		self.__y.append(torch.zeros((1, self.__input_size), requires_grad = False))

		for l in range(self.__depth):

			self.__y.append(torch.zeros((1, self.__hidden_size), requires_grad = False))

		self.__y.append(torch.zeros((1, self.__output_size), requires_grad = False))

	def train(self, datapoints):

		print(f'\n> training Hebb-MLP for {self.__epochs} epochs...')

		for e in range(self.__epochs):

			loss = 0.0

			self.__reset_Hebb()								# reset Hebbian terms.

			for i in tqdm(range(len(datapoints['X'])), desc = f'Training epoch {e+1}/{self.__epochs}'):

				self.__reset_units()						# reset units' actiavtions.

				self.optimizer.zero_grad()

				self.__y[0][0] = torch.tensor(datapoints['X'][i])	# input layer's activation.

				for l in range(self.__depth+1):

					# propagate input.

					self.__y[l+1][0] = F.tanh(
						self.__y[l].mm(self.__w[l] + torch.mul(self.__alpha[l], self.__Hebb[l]))
						)

					# update hebbian terms.

					pre_y = self.__y[l].view(self.__y[l].shape[1], self.__y[l].shape[0])
					post_y = self.__y[l+1].expand(self.__y[l].shape[1], self.__y[l+1].shape[1])

					self.__Hebb[l] = (1.0 - self.s1) * self.__Hebb[l] + self.s1 * (pre_y * post_y)

					torch.clamp(self.__alpha[l], min = self.alpha_min, max = self.alpha_max)

				# computing loss (MSE).

				loss += ((self.__y[-1][0] - torch.tensor(datapoints['Y'][i], requires_grad = False)) ** 2)

			loss /= len(datapoints['X'])

			if e % 10 == 0 and self.show_loss:
				print(f'epoch: {e}, loss: {loss.item()}')

			self.losses.append(loss.item())

			loss.backward()								# backpropagate.
			self.optimizer.step()

	def predict(self, datapoints):

		print(f'\n> testing Hebb-MLP...')

		loss = []

		self.__reset_Hebb()

		for i in tqdm(range(len(datapoints['X'])), desc = 'testing'):

			self.__reset_units()								# reset units' actiavtions.

			self.__y[0][0] = torch.tensor(datapoints['X'][i])	# input layer's activation.

			for l in range(self.__depth+1):						# propagate input.

				self.__y[l+1][0] = F.tanh(
					self.__y[l].mm(self.__w[l] + torch.mul(self.__alpha[l], self.__Hebb[l]))
					)

				# update hebbian terms.

				pre_y = self.__y[l].view(self.__y[l].shape[1], self.__y[l].shape[0])
				post_y = self.__y[l+1].expand(self.__y[l].shape[1], self.__y[l+1].shape[1])

				self.__Hebb[l] = (1.0 - self.s1) * self.__Hebb[l] + self.s1 * (pre_y * post_y)

				torch.clamp(self.__alpha[l], min = self.alpha_min, max = self.alpha_max)

			# computing loss (MSE).

			loss.append(((self.__y[-1][0] - torch.tensor(datapoints['Y'][i], requires_grad = False)) ** 2).tolist()[0])

		return loss

	def export_parameters(self, phase):

		__w = list(self.__w)
		__alpha = list(self.__alpha)

		_dict_export = {'w': {}, 'alpha': {}}

		for l in range(len(__w)):

			_dict_export['w'][l] = __w[l].tolist()
			_dict_export['alpha'][l] = __alpha[l].tolist()

		with open(f'../exported/Hebb-MLP_parameters-{phase}.pickle', 'wb') as file:
			pickle.dump(_dict_export, file)

	def load_parameters(self, parameters):

		loaded = pickle.load(open(parameters, 'rb'))

		for layer, weights in loaded['w'].items():
			self.__w.append(torch.tensor(weights))

		for layer, weights in loaded['alpha'].items():
			self.__alpha.append(torch.tensor(weights))

	def infer(self, datapoints):

		self.__reset_units()								# reset units' actiavtions.

		self.__y[0][0] = torch.tensor(datapoints['X'][i])	# input layer's activation.

		for l in range(self.__depth+1):						# propagate input.

			self.__y[l+1][0] = F.tanh(
				self.__y[l].mm(self.__w[l] + torch.mul(self.__alpha[l], self.__Hebb[l]))
				)

			# update hebbian terms.

			pre_y = self.__y[l].view(self.__y[l].shape[1], self.__y[l].shape[0])
			post_y = self.__y[l+1].expand(self.__y[l].shape[1], self.__y[l+1].shape[1])

			self.__Hebb[l] = (1.0 - self.s1) * self.__Hebb[l] + self.s1 * (pre_y * post_y)

		# computing loss (MSE).


		return self.__y[-1][0].tolist()[0]

