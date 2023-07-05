import torch, pickle
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm.auto import tqdm


class MLP:
    def __init__(self, args):

        self.__input_size = args.input_size
        self.__hidden_size = args.hidden_size
        self.__output_size = args.output_size
        self.__depth = args.depth  # number of hidden layers.

        self.__lr = args.lr  # learning rate.
        self.__epochs = args.epochs  # number of epochs.

        self.s1 = .01  # scaling factor for fixed weight and alpha.

        self.show_loss = args.show_loss

        self.__w = nn.ParameterList()  # fixed weights.
        self.__y = []  # units' activations.

        self.__init_ANN()

        self.losses = []

        self.__reset_units()

    def __init_ANN(self):

        self.__init_parameters()

        self.optimizer = torch.optim.Adam(list(self.__w), lr=self.__lr)

    def __init_parameters(self):

        # w/alpha/Hebb between input and 1st hidden layer.

        self.__w.append(nn.Parameter(
            self.s1 * torch.randn(
                (self.__input_size, self.__hidden_size), requires_grad=True)))

        # w/alpha/Hebb between hidden layers.

        for l in range(self.__depth - 1):
            self.__w.append(nn.Parameter(
                self.s1 * torch.randn(
                    (self.__hidden_size, self.__hidden_size), requires_grad=True)))

        # w/alpha/Hebb between last hidden layer and output.

        self.__w.append(nn.Parameter(
            self.s1 * torch.randn(
                (self.__hidden_size, self.__output_size), requires_grad=True)))

    def __reset_units(self):

        self.__y = []

        self.__y.append(torch.zeros((1, self.__input_size), requires_grad=False))

        for l in range(self.__depth):
            self.__y.append(torch.zeros((1, self.__hidden_size), requires_grad=False))

        self.__y.append(torch.zeros((1, self.__output_size), requires_grad=False))

    def train(self, datapoints):
        print(f'\n> training MLP for {self.__epochs} epochs...')

        epoch_loss = []
        for e in tqdm(range(self.__epochs), leave=True):
            exp_loss = []
            for exp in tqdm(datapoints, leave=False):
                obs_loss = []
                for inp, tgt in zip(exp['X'], exp['Y']):
                    loss = 0.0
                    self.__reset_units()  # reset units' activations.
                    self.optimizer.zero_grad()

                    self.__y[0][0] = torch.tensor(inp)  # input layer's activation.
                    for l in range(self.__depth):
                        # propagate input.
                        self.__y[l + 1][0] = F.tanh(self.__y[l].mm(self.__w[l]))
                    self.__y[-1][0] = self.__y[self.__depth].mm(self.__w[self.__depth])

                    # computing loss (MSE).
                    loss = (self.__y[-1][0] - torch.tensor(tgt, requires_grad=False)) ** 2
                    loss.backward()
                    self.optimizer.step()

                    obs_loss.append(loss.item())

                exp_loss.append(np.mean(obs_loss))
                # print('exp_loss', exp_loss[-1])

            epoch_loss.append(np.mean(exp_loss))

            if self.show_loss:
                print(f'epoch: {e}, loss: {epoch_loss[-1]}')

    def predict(self, datapoints):

        print(f'\n> testing MLP...')

        loss = []

        for exp in tqdm(datapoints):
            for inp, tgt in zip(exp['X'], exp['Y']):
                # self.__reset_units()  # reset units' activations.

                self.__y[0][0] = torch.tensor(inp)  # input layer's activation.

                for l in range(self.__depth):  # propagate input.

                    self.__y[l + 1][0] = F.tanh(self.__y[l].mm(self.__w[l]))

                self.__y[-1][0] = self.__y[1].mm(self.__w[1])

                # computing loss (MSE).

                loss.append(((self.__y[-1][0] - torch.tensor(tgt, requires_grad=False)) ** 2).tolist()[0])

        return loss

    def export_parameters(self, dataset, phase):

        __w = list(self.__w)

        _dict_export = {'w': {}}

        for l in range(len(__w)):
            _dict_export['w'][l] = __w[l].tolist()

        with open(f'../exported/MLP_parameters-{dataset}-{phase}.pickle', 'wb') as file:
            pickle.dump(_dict_export, file)

    def load_parameters(self, parameters):

        loaded = pickle.load(open(parameters, 'rb'))

        for layer, weights in loaded['w'].items():
            self.__w.append(torch.tensor(weights))

    def infer(self, values):
        self.__y[0][0] = torch.tensor(values)  # input layer's activation.

        for l in range(self.__depth):  # propagate input.

            self.__y[l + 1][0] = F.tanh(self.__y[l].mm(self.__w[l]))

        self.__y[-1][0] = self.__y[1].mm(self.__w[1])

        return self.__y[-1][0].tolist()[0]
