import random, pickle

def generate_data(n, l):

	data = {'X': [], 'Y': []}
	half_n = n // 2

	for _ in range(n):

		vector = [random.random() for _ in range(l)]
		data['X'].append(vector)

		if len(data['Y']) < half_n:
			data['Y'].append(1)
		else:
			data['Y'].append(0)

	zipped_data = list(zip(data['X'], data['Y']))
	random.shuffle(zipped_data)
	shuffled_X, shuffled_Y = zip(*zipped_data)

	data['X'] = list(shuffled_X)
	data['Y'] = list(shuffled_Y)

	with open('../exported/bin_classification_data.pickle', 'wb') as file:
		pickle.dump(data, file)

generate_data(50, 5)	# generate random binary classification training data.

