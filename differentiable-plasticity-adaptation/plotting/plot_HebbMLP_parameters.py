import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize = (10, 6))
grid = GridSpec(2, 2, figure = fig)

# Create subplots within the grid
ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, 0])
ax4 = fig.add_subplot(grid[1, 1])

colors = ['r', 'b', 'g', 'purple']

# pre training

with open('../exported/Hebb-MLP_parameters-before.pickle', 'rb') as file:
	params_before = pickle.load(file)

for layer, values in params_before['w'].items():
	ax1.hist(values, facecolor = colors[layer] , label = f'l {layer}', alpha = 0.5)

ax1.set_xlabel('w (pre-training)')

for layer, values in params_before['alpha'].items():
	ax2.hist(values, facecolor = colors[layer] , label = f'l {layer}', alpha = 0.5)

ax2.set_xlabel('alpha (pre-training)')

# post training

with open('../exported/Hebb-MLP_parameters-after.pickle', 'rb') as file:
	params_after = pickle.load(file)

for layer, values in params_after['w'].items():
	ax3.hist(values, facecolor = colors[layer] , label = f'l {layer}', alpha = 0.5)

ax3.set_xlabel('w (post-training)')

for layer, values in params_after['alpha'].items():
	ax4.hist(values, facecolor = colors[layer] , label = f'l {layer}', alpha = 0.5)

ax4.set_xlabel('alpha (post-training)')

fig.tight_layout()
plt.show()
plt.close()