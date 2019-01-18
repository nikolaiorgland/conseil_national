# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:15:35 2019

@author: silus
"""
import numpy as np
import networkx as nx
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
    """ Computes the ratio between average edge weight to nodes outside of community
    and average edge weights to nodes in the same community. 
    community_type      'party' for comparison using party association as communities
                        'modularity_max' for comparison using modularity-maximizing 
                        partitions.
    leg                 list of string with legislatures of interest
    years_of_leg        list of years to be analyzed in the legislatures. Must be same 
                        length as len(leg) or None to use network based on data of the 
                        whole legislature."""
    
    
    if community_type not in ['party','modularity_max']:
        raise TypeError("Not a valid community type")
            
    assert isinstance(leg, list)
    if years_of_leg:
        assert isinstance(years_of_leg, list)
        assert len(leg) == len(years_of_leg)
    
    adjacencies, node_indices = get_adjacencies_per_year(leg, years_of_leg)
    
    ret_comm_present = []
    ret_comm_size = []
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
        ret_comm_size.append(community_size)
        ret_comm_loyalty.append(community_loyalty)
        ret_node_loyalty.append(df)
        
    years = []
    if years_of_leg:
        for i,l in enumerate(leg):    
            for y in range(1,years_of_leg[i]+1):
                years.append(str(2007+(int(l)-48)*4+y))
    else:
        years = leg
                
    return ret_comm_present, ret_comm_size, ret_comm_loyalty, ret_node_loyalty, years

def centralities(leg,years_of_leg,cut_off):
    
    legislatures = []
    adjacencies, node_indices = get_adjacencies_per_year(leg, years_of_leg)
    
    for adjacency, node_list in zip(adjacencies, node_indices):
   
        name_with_party = assign_party_to_names('../data/Ratsmitglieder_1848_FR.csv', node_list)
    
        # Cut-off: Eliminate elements from the adjacency matrix below a certain treshold
        adjacency_mod = adjacency.copy()
        adjacency_mod[[adjacency_mod < cut_off]] = 0
    
        # Creation of networkx graph from adjacency
        G=nx.from_numpy_matrix(adjacency_mod)
    
        closeness_cent = nx.closeness_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        name_with_party = name_with_party.drop(columns='Counc_Id')
        name_with_party.insert(2,'Closeness centrality', list(closeness_cent.values()))
        name_with_party.insert(3, 'Betweenness centrality',list(betweenness_cent.values()))
        
        legislatures.append(name_with_party)
        
    years = []
    if years_of_leg:
        for i,l in enumerate(leg):
            for y in range(1,years_of_leg[i]+1):
                years.append(str(2007+(int(l)-48)*4+y))
    else:
        years = leg
    return legislatures, years


def compute_modularity(leg, years_of_leg, resolution=1):
    
    assert isinstance(leg, list)
    if years_of_leg:
        assert isinstance(years_of_leg, list)
        assert len(leg) == len(years_of_leg)
    
    evolution_modularity = []
    
    adjacencies, node_indices = get_adjacencies_per_year(leg, years_of_leg)
    
    for adjacency in adjacencies:
        partitions, modularity = detect_partitions(adjacency, resolution=resolution)
        evolution_modularity.append(modularity)
    
    years = []
    if years_of_leg:
        for i,l in enumerate(leg):
            for y in range(1,years_of_leg[i]+1):
                years.append(str(2007+(int(l)-48)*4+y))
    else:
        years = leg
    return evolution_modularity, years