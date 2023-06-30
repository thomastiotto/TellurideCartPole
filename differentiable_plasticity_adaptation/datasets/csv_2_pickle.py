import csv, os, pickle

dataset = '27s'

columns = ['time', 'angle', 'angleD', 'angleDD', 'angle_cos', 'angle_sin', 'position', 'positionD', 'positionDD', 'Q', 'u', 'target_position', 'target_equilibrium', 'L', 'u_-1']
net_input = ['angleD', 'angle_cos', 'angle_sin', 'position', 'positionD', 'target_position', 'target_equilibrium', 'L']

net_input_index = []

for i in range(len(columns)):

    if columns[i] in net_input:

        net_input_index.append(i)

net_output_index = columns.index('Q')

# === Training data =======================================================

print('> generating train data...')

_train_data = {'X': [], 'Y': []}

_csv_files = os.listdir(os.path.join('Train', f'Train-{dataset}'))

for file in _csv_files:

    _data_row_start = -1

    # Open the CSV file
    with open(os.path.join('Train', f'Train-{dataset}', file), 'r') as file:

        reader = csv.reader(file)

        for row in reader:

            if 'target_equilibrium' in row:
                _data_row_start = 1

            elif _data_row_start != -1 and row[0] != '':

                _feat_vec = []

                for index in net_input_index:

                    _feat_vec.append(float(row[index]))

                target = float(row[net_output_index])

                _train_data['X'].append(_feat_vec)
                _train_data['Y'].append(target)


# === Test data =======================================================

print('> generating test data...')

_test_data = {'X': [], 'Y': []}

_csv_files = os.listdir(os.path.join('Test', f'Test-{dataset}'))

for file in _csv_files:

    _data_row_start = -1

    # Open the CSV file
    with open(os.path.join('Test', f'Test-{dataset}', file), 'r') as file:

        reader = csv.reader(file)

        for row in reader:

            if 'target_equilibrium' in row:
                _data_row_start = 1

            elif _data_row_start != -1 and row[0] != '':

                _feat_vec = []

                for index in net_input_index:

                    _feat_vec.append(float(row[index]))

                target = float(row[net_output_index])

                _test_data['X'].append(_feat_vec)
                _test_data['Y'].append(target)

# === Export ==========================================================

print('> exporting data...')

with open(f'../datasets/Train-{dataset}-noise.pickle', 'wb') as file:
    pickle.dump(_train_data, file)

with open(f'../datasets/Test-{dataset}-noise.pickle', 'wb') as file:
    pickle.dump(_test_data, file)

