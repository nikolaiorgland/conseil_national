# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:00:06 2018

@author: silus
"""
import pandas as pd
import numpy as np

nodes = pd.read_csv('../data/node_idx_cluster.csv')
parties = pd.read_csv('../data/conseil_national_parties.csv')

parties = parties.drop_duplicates(keep='first')
parties['Name'] = parties['LastName']+' '+parties['Name'] 
parties = parties.drop(columns=['LastName'])

name_parties = nodes.join(parties.set_index('Name'), on='CouncillorName')
party_cluster_dict = {'SVP': [1], 'SP': [-1],'GPS':[-1],'CVP':[1],
                 'FDP-Liberale':[1],'EVP':[-1], 'glp':[-1],
                 'BDP':[1],'Lega':[1],'csp-ow':[1],'MCG':[1],
                 'PdA':[-1],'LDP':[1],'BastA':[-1],'CSPO':[1]}
party2cluster = pd.DataFrame(data=party_cluster_dict).T
party2cluster = party2cluster.rename(columns={0:'cluster'})
name_parties_cluster = name_parties.set_index('party').join(party2cluster)

clustering = name_parties_cluster.sort_values('node_idx')
np.save('clustering.npy', clustering['cluster'])
