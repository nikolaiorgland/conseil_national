# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:15:35 2019

@author: silus
"""
import numpy as np
import networkx as nx
import pandas as pd
import community as pylouvain
from load_and_preprocessing import get_adjacencies_per_year, assign_party_to_names
from helpers import make_signal, convert_dframecol_to_dict

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

def compute_community_loyalty(community_type, leg, years_of_leg):
    
    if community_type not in ['party','modularity_max']:
        raise TypeError("Not a valid community type")
            
    assert isinstance(leg, list)
    assert isinstance(years_of_leg, list)
    assert len(leg) == len(years_of_leg)
    
    adjacencies, node_indices = get_adjacencies_per_year(leg, years_of_leg)
    
    ret_comm_present = []
    ret_comm_loyalty = []
    ret_node_loyalty = []
    
    for adjacency, node_list in zip(adjacencies, node_indices):
        n_nodes = adjacency.shape[0]
        
        if community_type == 'modularity_max':
            community_labels, modularity = detect_partitions(adjacency, resolution=1)
        if community_type == 'party':
            name_with_party = assign_party_to_names('../data/Ratsmitglieder_1848_FR.csv', node_list)
            community_labels = convert_dframecol_to_dict(name_with_party, 'PartyAbbreviation')
            
        community_assignments = make_signal(n_nodes, community_labels)
        communities_present, community_size = np.unique(community_assignments, return_counts=True)
        node_loyalty = np.zeros(n_nodes)
        community_loyalty = np.zeros(len(communities_present))
        
        for i, comm in enumerate(communities_present):
            (nodes_in_comm,) = np.where(community_assignments == comm)
            edges = adjacency[nodes_in_comm,:]
            edges_in_comm = edges[:,nodes_in_comm]
            mask = np.ones(edges.shape[1], np.bool)
            mask[nodes_in_comm] = 0
            edges_to_outside = edges[:,mask]
            
            comm_total = np.sum(edges_in_comm)/2
            comm_total_outside = np.sum(edges_to_outside)/2
            community_loyalty[i] = (comm_total*(n_nodes-community_size[i]))/(comm_total_outside*community_size[i])
            node_loyalty[nodes_in_comm] = (np.sum(edges_in_comm, axis=1)/community_size[i])/(np.sum(edges_to_outside, axis=1)/(n_nodes-community_size[i]))
        
        df = node_list[['CouncillorName']]
        df.insert(1,'loyalty', node_loyalty)
        df.insert(2,'community', community_assignments)
        
        ret_comm_present.append(communities_present)
        ret_comm_loyalty.append(community_loyalty)
        ret_node_loyalty.append(df)
    return ret_comm_present, ret_comm_loyalty, ret_node_loyalty