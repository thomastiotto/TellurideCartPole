import csv, os, pickle

columns = ['time', 'angle', 'angleD', 'angleDD', 'angle_cos', 'angle_sin', 'position', 'positionD', 'positionDD', 'Q',
           'u', 'target_position', 'target_equilibrium', 'L', 'u_-1']
net_input = ['angleD', 'angle_cos', 'angle_sin', 'position', 'positionD', 'target_position', 'target_equilibrium']

net_input_index = []

for i in range(len(columns)):

    if columns[i] in net_input:
        net_input_index.append(i)

net_output_index = columns.index('Q')

dataset = 'CPS-17-02-2023-UpDown-Imitation'


def generate_data(split):
    print(f'> generating {split} data...')

    csv_files = os.listdir(
        f'../../SI_Toolkit_ASF/{dataset}/Recordings/{split}')

    experiments = []
    for file in csv_files:
        data = {'X': [], 'Y': []}

        data_row_start = -1

        # Open the CSV file
        with open(
                f'../../SI_Toolkit_ASF/{dataset}/Recordings/{split}/{file}',
                'r') as file:

            reader = csv.reader(file)

            for row in reader:

                if 'target_equilibrium' in row:
                    data_row_start = 1

                elif data_row_start != -1 and row[0] != '':

                    _feat_vec = []

                    for index in net_input_index:
                        _feat_vec.append(float(row[index]))

                    target = float(row[net_output_index])

                    data['X'].append(_feat_vec)
                    data['Y'].append(target)
        experiments.append(data)

    return experiments


def find_min_max(data):
    max = 0
    min = 0

    for d in data:
        for obs in d['X']:
            for o in obs:
                if o > max:
                    max = o
                if o < min:
                    min = o
    return min, max


def normalize_data(data, min, max):
    import numpy as np

    print('> normalizing data...')

    for d in data:
        for i, obs in enumerate(d['X']):
            d['X'][i] = list(2 * ((np.array(obs) - min) / (max - min)) - 1)

    return data


train_data = generate_data('Train')
test_data = generate_data('Test')
min, max = find_min_max(train_data + test_data)
train_data_norm = normalize_data(train_data, min, max)
test_data_norm = normalize_data(test_data, min, max)

# === Export ==========================================================

print('> exporting data...')

with open(f'../datasets/Train-{dataset}-noise.pickle', 'wb') as file:
    pickle.dump(train_data_norm, file)

with open(f'../datasets/Test-{dataset}-noise.pickle', 'wb') as file:
    pickle.dump(test_data_norm, file)
