# -*- coding: utf-8 -*-
"""
Created on Tue Jan 9 12:13:44 2019

@author: Diego
"""

import numpy as np
import pandas as pd
from load_and_preprocessing import assign_party_to_names, get_adjacencies_per_year
from helpers import get_lap_eigendecomp
# This function extracts the entries for a specific councillor from the full table that is sorted by date

def create_evolution_features_v1(leg, years_of_leg):
    
    assert isinstance(leg, list)
    assert isinstance(years_of_leg, list)
    assert len(leg) == len(years_of_leg)
    
    adjacencies, node_indices = get_adjacencies_per_year(leg, years_of_leg)
    
    output = []
    
    # Create plot for every legislature
    for adjacency, node_index in zip(adjacencies, node_indices):
        n_councillors = adjacency.shape[0]
        name_with_party = assign_party_to_names('../data/Ratsmitglieder_1848_FR.csv', node_index)
        eigenvals, eigenvectors = get_lap_eigendecomp(adjacency, lap_type='normalized', ret_eigval=True)

        parties_of_nodes = name_with_party['PartyAbbreviation'].values
        
        parties_represented, number_of_members =  np.unique(parties_of_nodes, return_counts=True)
        parties_orientations = []
        
        # Sum up values of 2nd eigenvector for all nodes belonging to a particular party
        # Normalize sign of eigenvector entries by making sure that right wing party UDC 
        # always has positive orientation
        flip_sign = False
        for party in parties_represented:
            party_orientation = np.mean(eigenvectors[parties_of_nodes== party, 1])
            if party == 'UDC':
                if party_orientation < 0:
                    flip_sign = True
            parties_orientations.append(party_orientation)
        
        if flip_sign:
            parties_orientations = -np.array(parties_orientations)
        else:
            parties_orientations = np.array(parties_orientations)
        
        # Percentage of seats held by a party
        parties_domination = number_of_members/n_councillors
        
        if len(parties_orientations) != len(parties_domination):
            raise ValueError("Not the same number of orientation as dominations")
            
        orientation_and_domination = np.concatenate((parties_orientations.reshape(1,-1),
                                                     parties_domination.reshape(1,-1)))
        out = pd.DataFrame(orientation_and_domination, columns=parties_represented)
        output.append(out)
    
    years = []
    for i,l in enumerate(leg):
        for y in range(1,years_of_leg[i]+1):
            years.append(str(2007+(int(l)-48)*4+y))
            
    return output, years
            
            
            
            
                                    
    
