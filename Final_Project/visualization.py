# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:00:22 2019

@author: silus
"""
import numpy as np
import pandas as pd
import networkx as nx
import community as pylouvain

def get_lap_eigendecomp(adjacency, lap_type='combinatorial', ret_eigval=False):
    """ Returns eigenvectors of graph laplacian that can be used for laplacian eigenmaps"""
    
    if not isinstance(adjacency, np.ndarray):
        raise TypeError("Adjacency must numpy array")
    if len(adjacency.shape) != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('Adjacency matrix has incorrect shape {}'.format(adjacency.shape))
    if max(adjacency.shape) > 1000:
        raise ("High dimensional adjacency detected. This function is not adapted for high dimensional arrays")
        
    if lap_type not in ['combinatorial', 'normalized']:
            print("Unknown Laplacian type. Using combinatorial...")
            lap_type = 'combinatorial'
        
    n_nodes = adjacency.shape[0]
    
    # Compute laplacian
    if lap_type == 'combinatorial':
        D = np.diag(np.ravel(np.sum(adjacency, axis=1)), k=0)
        L = D - adjacency
    else:
        dw = np.sum(adjacency, axis=1)
        d = np.power(dw, -0.5)
        D = np.diag(np.ravel(d), k=0)
        L = np.eye(n_nodes) - D @ adjacency @ D
    
    eigenvals, eigenvecs = np.linalg.eigh(L)
    
    if ret_eigval:
        return eigenvals, eigenvecs
    else:
        return eigenvecs
    
def label_to_numeric(node_index_with_labels, label_name, dictionary, ret_values=False):
    """ Converts any label of the nodes such as party or lobbying mandats into
    numerical values that can be plotted according to a dictionary that is passed
    as an argument
    node_index_with_label       pd.DataFrame were a row represents a node and
                                the columns their labels
    label_name                  String, name of the label/column or interest in 
                                node_index_with_label
                                Example: 'PartyAbbreviation'
    dicitionary                 dict object, that associates a num. values to a 
                                label value
                                Example: {'UDC':1,'PS':-1}
    returns:
    node_index_with_label_num   pd.DataFrame identical to node_index_with label
                                    but with numerical values in label_name column"""
    
    if not isinstance(node_index_with_labels, pd.DataFrame):
        raise TypeError("Data must be in form of a pd.DataFrame. {0} detected"
                        .format(type(node_index_with_labels)))
    if not label_name in node_index_with_labels.columns:
        raise KeyError("Specified label was not found in data")
        
    node_index_with_labels_num = node_index_with_labels.replace({label_name: dictionary})
    
    if ret_values:
        unique_vals = node_index_with_labels[label_name].drop_duplicates(keep='first').values
        return node_index_with_labels_num, unique_vals
    else:
        return node_index_with_labels_num
    
def detect_partitions(adjacency, resolution=1.0):
    """ Detects partitions based on the Louvain method. Also returns network
    modularity
    
    adjacency       Numpy array
    resolution      float; can be used to tune resolution of Louvain algorithm.
                    set lower to get more communities
    
    returns:
    partition       dict with structure {node_id1:partition_id, node_id2:partition_id,...}
    modularity      float. Calculated modularity of network
    """
    
    if not isinstance(adjacency, np.ndarray):
        raise TypeError("Wrong array format. Adjacency matrix must be numpy.ndarray")
        
    graph = nx.from_numpy_array(adjacency)
    partition = pylouvain.best_partition(graph, resolution=resolution)
    modularity = pylouvain.modularity(partition, graph)
    
    return partition, modularity

def make_signal(n_nodes, dictionary):
    
    label = np.zeros(n_nodes)
    for i in range(n_nodes):
        label[i] = dictionary[i]
    
    return label
        
    

    