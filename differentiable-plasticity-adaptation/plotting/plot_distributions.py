# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:23:44 2023

@author: RAX17
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('../datasets/normal_data_train.pickle', 'rb') as file:
    training_samples = pickle.load(file)
    
training_samples['X'] = np.array(training_samples['X'])
training_samples['Y'] = np.array(training_samples['Y'])

print(len(training_samples['X']))

with open('../datasets/normal_data_test.pickle', 'rb') as file:
    test_samples = pickle.load(file)

test_samples['X'] = np.array(test_samples['X'])
test_samples['Y'] = np.array(test_samples['Y'])

print(len(test_samples['X']))

train_v_class1 = training_samples["X"][training_samples["Y"]==0]
train_v_class2 = training_samples["X"][training_samples["Y"]==1]
test_v_class1 = test_samples["X"][test_samples["Y"]==0]
test_v_class2 = test_samples["X"][test_samples["Y"]==1]


plt.figure()
plt.title("Data distribution")
plt.scatter(train_v_class1[:,0], train_v_class1[:,1], alpha = 0.8)
plt.scatter(train_v_class2[:,0], train_v_class2[:,1], alpha = 0.8)
plt.scatter(test_v_class1[:,0], test_v_class1[:,1], alpha = 0.4)
plt.scatter(test_v_class2[:,0], test_v_class2[:,1], alpha = 0.4)
plt.legend(('Train Class 1','Train Class 2','Test Class 1','Test Class 2'))

plt.show()
plt.close()