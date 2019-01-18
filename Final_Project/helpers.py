# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:00:22 2019

@author: silus
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from load_and_preprocessing import get_adjacencies_per_year, assign_party_to_names

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
    
def convert_dframecol_to_dict(df, label_column_name):
    values = df[label_column_name].values
    keys = np.arange(len(values))
    return dict(zip(keys, values))

def make_signal(n_nodes, dictionary):
    # Initialize as list to accept strings also
    label = []
    for i in range(n_nodes):
        label.append(dictionary[i])
    
    return np.array(label)
    
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

def visualize_modularity(modularity_evolution, years):
    # Use load_data_and_filter_members to create adjacency for each year seperately

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(years, modularity_evolution, linewidth=1.5, color='k')
    ax1.set_xlabel("Year",fontsize=14)
    ax1.set_ylabel("Modularity",fontsize=14)
    ax1.set_ylim([0.2,0.45])
    ax1.set_yticks([0.2, 0.3, 0.35, 0.4, 0.45])
    ax1.set_yticklabels(["0", "0.3", "0.35", "0.4","0.45"], fontsize=13)
    ax1.set_xticks(years[0::2])
    ax1.set_xticklabels(years[0::2], fontsize=14)
    ax1.grid(True)
    
    plt.show()
    fig.savefig('figures/modularity_evolution.png', dpi=700, bbox_inches = "tight")
        

def visualize_node_loyalty(node_loyalty, padding=10, ):
    
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
    
def visualize_party_orientation(party_evolution_df, years, party_color_map):
    party_to_be_plotted = list(party_color_map.keys())
    fig, ax1 = plt.subplots(figsize=(14, 5))
    max_val = 0
    for party in party_to_be_plotted:
        orientation = []
        domination = []
        
        for values_per_year in party_evolution_df:
            values_party = values_per_year[party].values
            orientation.append(values_party[0])
            if abs(values_party[0]) > max_val:
                max_val = abs(values_party[0]) 
            domination.append(values_party[1])
        
        ax1.plot(10/0.12*np.array(orientation), years, c=party_color_map[party])
        
    fs = 15
    ax1.text(-9.5, 5, 'PSS', color=party_color_map['PSS'], fontsize=fs, fontweight='semibold')  
    ax1.text(-7.5, 5, 'PES', color=party_color_map['PES'], fontsize=fs, fontweight='semibold')
    ax1.text(-2.5, 5, 'pvl', color=party_color_map['pvl'], fontsize=fs, fontweight='semibold')
    ax1.text(0.5, 5, 'PDC', color=party_color_map['PDC'], fontsize=fs, fontweight='semibold')
    ax1.text(2, 5, 'PBD', color=party_color_map['PBD'], fontsize=fs, fontweight='semibold')
    ax1.text(4, 5, 'PLR', color=party_color_map['PLR'], fontsize=fs, fontweight='semibold')
    ax1.text(5.5, 5, 'UDC', color=party_color_map['UDC'], fontsize=fs, fontweight='semibold')  
    ax1.grid(True, axis='x')
    ax1.set_yticks(years[0::2])
    ax1.set_yticklabels(years[0::2], fontsize=14)
    ax1.set_xticks([-10,-5,0,5,10])
    ax1.set_xticklabels([-10,-5,0,5,10], fontsize=14)
    ax1.set_xlabel('Avg. Fiedler vector components of members',fontsize=14)
    fig.savefig('figures/party_orientation.png', dpi=800, bbox_inches='tight')
    
    
def visualize_community_isolation(communities, community_size, community_loyalty, years, party_color_map, padding=10):
    comm_found = set()
    for c in communities:
        for p in c:
            comm_found.add(p)
    communities = np.array(communities)
    community_loyalty = np.array(community_loyalty)
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    for comm in comm_found:
        plot_comm = []
        for i, c in enumerate(communities):
            (idx,) = np.where(c == comm)
            if len(idx)==0:
                break
            plot_comm.append(community_loyalty[i][idx])
        if len(plot_comm) == len(years):
            ax1.plot(years, plot_comm, c=party_color_map[comm], linewidth=1.5)
            ax1.text(len(years)-1, plot_comm[-1], comm, fontsize=12, color=party_color_map[comm])
    ax1.set_xlabel('Year',fontsize=14)
    ax1.set_ylabel('Party isolation',fontsize=14)
    ax1.set_xticks(years[0::2])
    ax1.set_xticklabels(years[0::2], fontsize=14)
    ax1.grid(True, axis='y')
    fig.show()
    fig.savefig('figures/community_isolation.png',dpi=800, bbox_inches='tight')
        
    
    
    #fig, ax1 = plt.subplots(figsize=(15, 5))
    #for i in range(communities.shape[1]):
        
    
    