import numpy as np
import matplotlib.pyplot as plt
import pickle, random

# Number of points in each circle
num_points = 200

# Inner circle parameters
inner_radius = 2
outer_radius = 4

# === class 1 ==================================================

inner_x_A_center = 0
inner_y_A_center = 0

outer_x_A_center = 0
outer_y_A_center = 0

# Generate random points for the inner circle
inner_theta = np.random.uniform(0, 2 * np.pi, num_points)
inner_radius_offset = np.random.uniform(0, inner_radius, num_points)
inner_x_A = inner_x_A_center + (inner_radius - inner_radius_offset) * np.cos(inner_theta)
inner_y_A = inner_y_A_center + (inner_radius - inner_radius_offset) * np.sin(inner_theta)

# Generate random points for the outer circle
outer_theta = np.random.uniform(0, 2 * np.pi, num_points)
outer_radius_offset = np.random.uniform(inner_radius, outer_radius, num_points)
outer_x_A = outer_x_A_center + (outer_radius + inner_radius_offset) * np.cos(outer_theta)
outer_y_A = outer_y_A_center + (outer_radius + inner_radius_offset) * np.sin(outer_theta)

# Plotting
plt.scatter(inner_x_A, inner_y_A, color='blue')
plt.scatter(outer_x_A, outer_y_A, color='darkblue')

# === class 2 ==================================================

inner_x_B_center = 8
inner_y_B_center = 8

outer_x_B_center = 8
outer_y_B_center = 8

# Generate random points for the inner circle
inner_theta = np.random.uniform(0, 2 * np.pi, num_points)
inner_radius_offset = np.random.uniform(0, inner_radius, num_points)
inner_x_B = inner_x_B_center + (inner_radius - inner_radius_offset) * np.cos(inner_theta)
inner_y_B = inner_y_B_center + (inner_radius - inner_radius_offset) * np.sin(inner_theta)

# Generate random points for the outer circle
outer_theta = np.random.uniform(0, 2 * np.pi, num_points)
outer_radius_offset = np.random.uniform(inner_radius, outer_radius, num_points)
outer_x_B = outer_x_B_center + (outer_radius + inner_radius_offset) * np.cos(outer_theta)
outer_y_B = outer_y_B_center + (outer_radius + inner_radius_offset) * np.sin(outer_theta)

# Plotting
plt.scatter(inner_x_B, inner_y_B, color='red')
plt.scatter(outer_x_B, outer_y_B, color='darkred')
plt.axis('equal')
plt.show()
plt.close()

# === export train data ===================================================

train_data = {'X': [], 'Y': []}

for i in range(len(inner_x_A)):

	train_data['X'].append([inner_x_A[i], inner_y_A[i]])
	train_data['Y'].append(0)

for i in range(len(inner_x_B)):

	train_data['X'].append([inner_x_B[i], inner_y_B[i]])
	train_data['Y'].append(1)

zipped_data = list(zip(train_data['X'], train_data['Y']))
random.shuffle(zipped_data)
shuffled_X, shuffled_Y = zip(*zipped_data)

train_data['X'] = list(shuffled_X)
train_data['Y'] = list(shuffled_Y)

with open('../datasets/cocentric_circles-train.pickle', 'wb') as file:
	pickle.dump(train_data, file)

# === export test data ===================================================

test_data = {'X': [], 'Y': []}

for i in range(len(outer_x_A)):

	test_data['X'].append([outer_x_A[i], outer_y_A[i]])
	test_data['Y'].append(0)

for i in range(len(outer_x_B)):

	test_data['X'].append([outer_x_B[i], outer_y_B[i]])
	test_data['Y'].append(1)

zipped_data = list(zip(test_data['X'], test_data['Y']))
random.shuffle(zipped_data)
shuffled_X, shuffled_Y = zip(*zipped_data)

test_data['X'] = list(shuffled_X)
test_data['Y'] = list(shuffled_Y)

with open('../datasets/cocentric_circles-test.pickle', 'wb') as file:
	pickle.dump(test_data, file)
