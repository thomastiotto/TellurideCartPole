import numpy as np
import matplotlib.pyplot as plt
import math, pickle

sigma = 0.5
mu_x1 = 0.5
mu_x2 = 2.5
mu_xte = 6.75
x_1_tr = np.random.normal(mu_x1, sigma, 100)
x_2_tr = np.random.normal(mu_x2, sigma, 100)
x_te = np.random.normal(mu_xte, sigma, 100)

mu_y = 0.5
y_1_tr = np.random.normal(mu_y, sigma, 100)
y_2_tr = np.random.normal(mu_y, sigma, 100)
y_te = np.random.normal(mu_y, sigma, 100)

l_tr1 = []
l_tr2 = []
l_te = []

for i in range(len(x_1_tr)):

	distance = math.sqrt((mu_x1-x_1_tr[i])**2 + (mu_y - y_1_tr[i])**2)

	l_tr1.append(distance)

for i in range(len(x_2_tr)):

	distance = math.sqrt((mu_x2-x_2_tr[i])**2 + (mu_y - y_2_tr[i])**2)

	l_tr2.append(distance)

for i in range(len(x_te)):

	distance = math.sqrt((mu_xte-x_te[i])**2 + (mu_y - y_te[i])**2)

	l_te.append(distance)

_dict_training = {'X': list(zip(x_1_tr,y_1_tr)) + list(zip(x_2_tr,y_2_tr)), 'Y': l_tr1 + l_tr2}
_dict_test = {'X': list(zip(x_te,y_te)), 'Y': l_te}

with open('../datasets/shifted_distribution_centroid-train.pickle', 'wb') as file:
	pickle.dump(_dict_training, file)

with open('../datasets/shifted_distribution_centroid-test.pickle', 'wb') as file:
	pickle.dump(_dict_test, file)

# plt.scatter([x_1_tr, x_2_tr, x_te], [y_1_tr, y_2_tr, y_te])
# plt.scatter([0.5], [0.5], c = 'r')
# plt.scatter([2.5], [0.5], c = 'r')
# plt.scatter([6.75], [0.5], c = 'r')
# plt.show()
