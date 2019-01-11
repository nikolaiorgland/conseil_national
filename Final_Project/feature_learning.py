# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:53:27 2019

@author: silus
"""

import numpy as np
from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def get_covotation_matrix(dataframe):
    
    # Number of nodes in the graph
    n_nodes = dataframe.shape[0]
    half = int(n_nodes/2)

    # Covotation matrix. Entry at i,j is how many times node i and j voted the 
    # same thing
    covotation_matrix = np.zeros((half, n_nodes-half))
    
    for i, a in dataframe.iloc[0:half,:].iterrows():
        for j, b in dataframe.iloc[half:,:].iterrows():
            covotation_matrix[i,j-half] = np.sum(a == b)
            
    return np.log(covotation_matrix)
    
def get_data_split_indices(n_samples, k_fold=5, seed=567):
    """ Create train and test sets for k-fold cross validation. Returns them as
    a train and test covotation matrix """
    np.random.seed(seed)
    idx_interval = int(n_samples/k_fold)
    index = np.random.permutation(n_samples)
    split_index = [index[k*idx_interval:(k+1)*idx_interval] for k in range(k_fold)]
    return np.array(split_index)

def compute_rmse(X,W,H):
    dx = (X - W @ H)**2
    return np.sqrt(dx.mean())

def cross_validation(dataframe, lambdas, num_features, k_fold=5):
    
    n_samples = dataframe.shape[1]
    
    cv_index = get_data_split_indices(n_samples, k_fold)
    losses_tr = np.zeros((len(num_features), len(lambdas)))
    losses_te = np.zeros((len(num_features), len(lambdas)))
    for idx_nfeat, n_feat in enumerate(num_features):
        for idx_lambda, lambda_ in enumerate(lambdas):
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for k in range(k_fold):
                test_columns = cv_index[k]
                train_columns = cv_index[~(np.arange(cv_index.shape[0]) == k)].flatten()
                
                covot_test = get_covotation_matrix(dataframe.iloc[:,test_columns])
                covot_train = get_covotation_matrix(dataframe.iloc[:,train_columns])
                
                print("Matrix factorization with {0} features and lambda = {1}".format(n_feat, lambda_))
                model = NMF(n_components=n_feat, init='random', alpha=lambda_, l1_ratio=0.0, max_iter=200)
                W = model.fit_transform(covot_train)
                H = model.components_
                print("Finished in {0} iterations".format(model.n_iter_))
                rmse_tr_tmp.append(compute_rmse(covot_train, W, H))
                rmse_te_tmp.append(compute_rmse(covot_test, W, H))
            losses_tr[idx_nfeat, idx_lambda] = np.mean(rmse_tr_tmp)
            losses_te[idx_nfeat, idx_lambda] = np.mean(rmse_te_tmp)
    return losses_tr, losses_te

def evaluate_optimal_solution(dataframe, losses_te, lambdas, num_features):
    covot = get_covotation_matrix(dataframe)
    l_min = np.argmin(losses_te, axis=1)
    n_feat_min = np.argmin(l_min)
    
    plt.figure()
    ax = sns.heatmap(losses_te, linewidth=0.6, cmap='RdYlGn')
    plt.tight_layout()  
    
    lambda_opt = lambdas[l_min[n_feat_min]]
    n_feat_opt = num_features[n_feat_min]
    print("The smallest loss was acheived with lambda = {0} and k = {1}".format(lambda_opt, n_feat_opt))
    
    model = NMF(n_components=n_feat_opt, init='random', alpha=lambda_opt)
    W = model.fit_transform(covot)
    H = model.components_
    return W, H

def visual_comparison(dataframe, W, H):
    covot = get_covotation_matrix(dataframe)
    loss = compute_rmse(covot, W, H)
    print("Loss is {0}".format(loss))
    plt.figure(figsize=(12,6))
    plt.subplot(211)
    ax = sns.heatmap(covot, cmap='RdYlGn')
    plt.title('Original covotation matrix')
    plt.subplot(212)
    ax2 = sns.heatmap(W @ H, cmap='RdYlGn')
    plt.title('WH factorization')
                

            
        
            
        
        
        
    