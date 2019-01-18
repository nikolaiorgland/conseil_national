# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:53:27 2019

@author: silus
"""
# Basic imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp

# sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# ConvNN
from lib import models, graph, coarsening, utils

def do_pca(X_train,n_comps=20, plot_expl_var=False):
    pca = PCA(n_components=n_comps)
    X_transf = pca.fit_transform(X_train)
    expl_var_ratio = pca.explained_variance_ratio_
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(np.cumsum(expl_var_ratio))
    fig.show()
    return X_transf
    
    
def fit_svm(X_train, y_train, kernel, hyperparam, k_fold=5):
    """
    X_train     Training data (n_samples x d_dimensions)
    y_train     Labels
    kernel      String; Kernel used for svm
    hyperparam  ndarray; values of the regularizer that are used
    k_fold      K-Fold cross-validation K parameter
    """
    
    assert len(hyperparam.shape) == 1
    assert X_train.shape[0] == len(y_train)
    
    if not isinstance(hyperparam, np.ndarray):
        raise TypeError("Hyperparameters need to be passed as numpy.ndarray")
    
    if len(hyperparam) > 100:
        raise Warning("Large number of hyperparameters detected. This might take a while")
    
    if kernel not in ['linear','rbf','polynomial','sigmoid']:
        raise Warning("Unknown kernel, defaulting to linear")
        kernel = 'linear'
    
    mean_accuracy = []
    accuracy_variance = []
    
    for c in hyperparam:
        clf = svm.SVC(kernel='linear', C=c)
        scores = cross_val_score(clf, X_train, y_train, cv=k_fold)
        mean = scores.mean()
        stdev = scores.std()
        variance = stdev**2
        mean_accuracy.append(mean)
        accuracy_variance.append(variance)
        
        print("Accuracy: %0.2f (+/- %0.2f)" % (mean, stdev * 2))
    
    return np.array(mean_accuracy), np.array(accuracy_variance)

def fit_log_regression(X, y, cs, k_fold=5):
    """
    X_train     Training data (n_samples x d_dimensions)
    y_train     Labels
    kernel      String; Kernel used for svm
    hyperparam  ndarray; values of the regularizer that are used
    k_fold      K-Fold cross-validation K parameter
    """
        
    assert X.shape[0] == len(y)
    
    clf = LogisticRegressionCV(cv=k_fold, Cs=cs, random_state=0, class_weight='balanced', multi_class='multinomial').fit(X, y)
    a = clf.score(X, y)     
    print("Accuracy: %0.2f" % (a))
    
    return a


def split_test_train_for_cv(n_samples, k_fold=5, seed=567):
    """ Create train and test sets for k-fold cross validation. Returns them as
    a train and test covotation matrix """
    #np.random.seed(seed)
    idx_interval = int(n_samples/k_fold)
    index = np.random.permutation(n_samples)
    split_index = [index[k*idx_interval:(k+1)*idx_interval] for k in range(k_fold)]
    return np.array(split_index)
    
def cross_validate_convNN(X, y, adjacency, name_param, value_param, k, num_levels=5):
    
    split_index = split_test_train_for_cv(X.shape[0], k_fold=k)
    graphs, perm = coarsening.coarsen(sp.csr_matrix(adjacency.astype(np.float32)), 
                                      levels=num_levels, self_connections=False)
    
    accuracy = []
    loss = []
    for param_val in value_param:
        accuracy_param = []
        loss_param = []
        for k_ in range(k):
            test_samples = split_index[k_]
            train_samples = split_index[~(np.arange(split_index.shape[0]) == k_)].flatten()
            
            X_train = X[train_samples]
            X_test = X[test_samples]
            y_train = y[train_samples]
            y_test = y[test_samples]
            
            X_train = coarsening.perm_data(X_train, perm)
            X_test = coarsening.perm_data(X_test, perm)
            n_train = X_train.shape[0]
            
            L = [graph.laplacian(A, normalized=True) for A in graphs]
            
            # Conv NN parameters
            params = dict()
            params['dir_name']       = 'demo'
            params['num_epochs']     = 30
            params['batch_size']     = 30
            params['eval_frequency'] = 30
            
            # Building blocks.
            params['filter']         = 'chebyshev5'
            params['brelu']          = 'b1relu'
            params['pool']           = 'apool1'
            
            # Number of classes.
            C = y.max() + 1
            assert C == np.unique(y).size
            
            # Architecture.
            params['F']              = [4, 8]  # Number of graph convolutional filters.
            params['K']              = [3, 3]  # Polynomial orders.
            params['p']              = [2, 8]    # Pooling sizes.
            params['M']              = [256, C]  # Output dimensionality of fully connected layers.
            
            # Optimization.
            params['regularization'] = 4e-5
            params['dropout']        = 1
            params['learning_rate']  = 3e-3
            params['decay_rate']     = 0.9
            params['momentum']       = 0.8
            params['decay_steps']    = n_train / params['batch_size']
            params[name_param]       = param_val
            
            model = models.cgcnn(L, **params)
            test_acc, train_loss, t_step = model.fit(X_train, y_train, X_test, y_test)
            accuracy_param.append([max(test_acc), np.mean(test_acc)])
            loss_param.append([max(train_loss), np.mean(train_loss)])
        print(np.array(accuracy_param))
        pm = np.mean(np.array(accuracy_param), axis=0)
        pl = np.mean(np.array(loss_param), axis=0)
        print("IIIII Accuracy: %0.2f (max) %0.2f (mean) Loss: %0.2f (max) %0.2f (mean)" 
              % (pm[0], pm[1], pl[0], pl[1]))
        accuracy.append(pm)
        loss.append(pl)
    return accuracy, loss
            

    
    

                

            
        
            
        
        
        
    