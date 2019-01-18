# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:00:22 2019

@author: silus
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from network_analysis import detect_partitions
from load_and_preprocessing import load_data_and_filter_members, get_adjacencies_per_year

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


def make_signal(n_nodes, dictionary):
    
    # Initialize as list to accept strings also
    label = []
    for i in range(n_nodes):
        label.append(dictionary[i])
    
    return np.array(label)

def convert_dframecol_to_dict(df, label_column_name):
    values = df[label_column_name].values
    keys = np.arange(len(values))
    return dict(zip(keys, values))

def visualize_modularity(resolution):
    # Use load_data_and_filter_members to create adjacency for each year seperately
    
    legis = ['48','49','50']
    duration = [4,4,3]
    evolution_modularity = []
    
    adjacencies = get_adjacencies_per_year(legis, duration)
    
    for adjacency in adjacencies:
        partitions, modularity = detect_partitions(adjacency, resolution=resolution)
        evolution_modularity.append(modularity)

    
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]    

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax.plot(years, evolution_modularity, 'linewidth'=0.5)
    ax.set_xlabel("Year",fontsize=12)
    ax.set_ylabel("Modularity",fontsize=12)
    ax.set_ylim([0,0.4])
    plt.show()
    fig.savefig('modularity_evolution.png', dpi=300, bbox_inches = "tight")
        
    return evolution_modularity   

def visualize_node_loyalty(node_loyalty, padding=10):
    
    assert isinstance(node_loyalty, pd.DataFrame)
    
    nodes_comm_loyalty = node_loyalty[['loyalty','community']].values
    comm_present, comm_size = np.unique(nodes_comm_loyalty[:,1], return_counts=True)
    
    fig, ax1 = plt.subplots(figsize=(15, 5))
    x_ticks = []
    for i, comm in enumerate(comm_present):
        # Don't plot tiny parties
        if comm_size[i] < 5:
            continue
        
        (rows,) = np.where(nodes_comm_loyalty[:,1] == comm)
        plot_this = nodes_comm_loyalty[rows,0]
        plot_this.sort()
        
        x = np.arange(np.sum(comm_size[:i])+(i+1)*padding, np.sum(comm_size[:i])+(i+1)*padding+comm_size[i])
        x_ticks.append([np.sum(comm_size[:i])+(i+1)*padding + comm_size[i]/2, comm])
        ax1.bar(x, plot_this, width=0.7, linewidth=0)
     
    x_ticks = np.array(x_ticks)
    ax1.set_ylabel(r'$\frac{\Sigma_{j \subset party} W_{ij}}{\Sigma_{j \subset\not party} W_{ij}}$', fontsize=18)
    ax1.set_xticks(x_ticks[:,0].astype(float))
    ax1.set_xticklabels(x_ticks[:,1], fontsize=14)
    fig.show()
    fig.savefig('figures/node_loyalty.png', dpi=600, bbox_inches = "tight")
    
#def visualize_party_isolation(party_loyalty, padding=10):
    
    