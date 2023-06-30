# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:24:23 2023

@author: RAX17
"""

import random, pickle
import matplotlib.pyplot as plt
import numpy as np



def generate_data_from_normal(n, means, covs, verbose=False, save=False):
    
    """
    
    Parameters
    ----------
    n : number of samples from a multivariate gaussian.
    mean : list of the mean vectors of the multivariate gaussian.
    cov : list covariance matrix for the multivariate gaussian.
    
    Returns
    -------
    data : structure containing the dataset.
    
    """
    
   
    data = {'X': [], 'Y': []}
    half_n = n // 2
    
    n_dim = len(means[0])
    
    
    class_1_vector = []
    class_2_vector = []

    for _ in range(n):

            if len(data['Y']) < half_n:
                vector = np.random.multivariate_normal(means[0], covs[0]) 
                data['X'].append(vector)
                class_1_vector.append(vector)
                data['Y'].append(0)
            else:
                vector = np.random.multivariate_normal(means[1], covs[1])
                data['X'].append(vector)
                class_2_vector.append(vector)
                data['Y'].append(1)

    zipped_data = list(zip(data['X'], data['Y']))
    random.shuffle(zipped_data)
    shuffled_X, shuffled_Y = zip(*zipped_data)

    data['X'] = list(shuffled_X)
    data['Y'] = list(shuffled_Y)
    
    if n_dim == 2 and verbose:
        plt.figure()
        plt.title("Data distribution")
        plt.scatter(np.array(class_1_vector)[:,0], np.array(class_1_vector)[:,1])
        plt.scatter(np.array(class_2_vector)[:,0], np.array(class_2_vector)[:,1])

    # with open('normal_data_train.pickle', 'wb') as file:
    with open('../exported/normal_data_test.pickle', 'wb') as file:
        pickle.dump(data, file)

    return data

n = 1000
# means = [[0,1],[1,0]] #train
means = [[0.5,1],[1.5,0]] #test
covs = [np.diag([0.1,1]),np.diag([0.1,1])]

generate_data_from_normal(n,means,covs,True,True)