# -*- coding: utf-8 -*-
"""
Created on Tue Jan 9 12:13:44 2019

@author: Diego
"""

import numpy as np
import pandas as pd
from load_and_preprocessing import load_data_and_filter_members, assign_party_to_names
from visualization import get_lap_eigendecomp
# This function extracts the entries for a specific councillor from the full table that is sorted by date

def create_evolution_features(time_span,party_map):
    years_of_leg=[1,2,3,4]
    X_graph=[]
    Domination=[]
    cover=len(time_span)*4
    Y_graph=np.arange(2008,2008+cover, 1)
    for leg in time_span:
        print("Legislature: {0}".format(leg))
        for year in years_of_leg:
            print("Year: {0}".format(year))
            adjacency, node_index, sum_na_per_row = load_data_and_filter_members('../data/abdb-de-all-affairs-'+leg+'-0.csv',filter_method='number_NA',cutoff=10,ret_transf=False,leg=leg,year_leg=year)
                                                                             
            name_with_party = assign_party_to_names('../data/Ratsmitglieder_1848_FR.csv', node_index)
            eigenvals, eigenvectors = get_lap_eigendecomp(adjacency, lap_type='normalized', ret_eigval=True)

            number_of_members = dict.fromkeys(party_map, 0 )
            party_orientation= dict.fromkeys(party_map, 0 )
            party_domination= dict.fromkeys(party_map, 0 )

            for index,row in name_with_party.iterrows():
                Pty_abrev=row['PartyAbbreviation']
                if (not Pty_abrev!=Pty_abrev and Pty_abrev in party_map):
                    number_of_members[Pty_abrev]=number_of_members[Pty_abrev]+1
                    party_orientation[Pty_abrev]=party_orientation[Pty_abrev]+eigenvectors[index,1]
            x_graph=np.asarray(list(party_orientation.values()))/np.asarray(list(number_of_members.values()))
            x_graph_dic=dict(zip(party_map.keys(), zip(x_graph.tolist())))
                    
            if x_graph_dic['UDC'][0]<0:
                x_graph_dic=dict(zip(party_map.keys(), zip((-x_graph).tolist())))
            X_graph.append(x_graph_dic)
        for i in party_domination:
            party_domination[i] = float(number_of_members[i]/200)
            Domination.append(party_domination)
    return X_graph,Y_graph,Domination

    
                                    
    
