# -*- coding: utf-8 -*-
"""
Created on Tue Jan 1 12:13:44 2019

@author: silvan
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def assign_val_to_vote(data):
    counc_votes = data.copy()
    nbr_votes = counc_votes.shape[0]
    counc_votes.insert(len(counc_votes.columns),'value',np.zeros(nbr_votes))
    counc_votes.loc[counc_votes.CouncillorYes.astype(int) == 1,'value'] = 1
    counc_votes.loc[counc_votes.CouncillorNo.astype(int) == 1,'value'] = 0
    counc_votes.loc[counc_votes.CouncillorPresident.astype(int) == 1,'value'] = np.nan
    counc_votes.loc[(counc_votes.CouncillorAbstain.astype(int) == 1) | 
                        (counc_votes.CouncillorExcused.astype(int) == 1) | 
                        (counc_votes.CouncillorNotParticipated.astype(int) == 1),'value'] = 0.5
    
    return counc_votes

# This function extracts the entries for a specific councillor from the full table that is sorted by date

def extract_feature_vec(val_votes, councId, complete_feat_idx):
    # Look for the votes of one particular councillor and keep values and affair Ids
    feature_vec = val_votes.loc[val_votes.CouncillorId == councId, :]
    feature_vec = feature_vec.drop(columns=['CouncillorId'])
    
    # For multiple votes with the same Id, only keep the deciding vote (the first one in the df)
    feature_vec = feature_vec[~feature_vec.index.duplicated(keep='first')]
    feature_vec = complete_feat_idx.join(feature_vec)
    
    # Rename the value column to have the councillor Id as name for identification later on
    feature_vec = feature_vec.rename(columns={'value':str(councId)})
    return feature_vec


def load_data_and_filter_members(datapath, start_date=None, end_date=None, filter_method='number_NA', cutoff=10, ret_transf=False):
    """ Loads a dataset (whose path is given in the datapath argument) for eg. a particular legislature.
    Further, the function filters out councillors based on the amount of NAs present in their feature vector.
    The exact method is chosen in filter_method. The options available are 'number_NA' or 'number_nodes'.
    datapath:               path to data csv. Example: '../data/abdb-de-all-affairs-50-0_new.csv'
    start_date, end date:   Start and end of period from which the votes should be analyzed
                            Format: dd/mm/yyyy
    filter_method:          Choice of filtering method 
                            Option 'number_NA': All councillors with more NAs in their feature vector
                                        than the number specified in cutoff are removed
                            Option 'number_nodes': The first N councillors with the least amount of NAs are kept, where N = cutoff
    cutoff:                 Integer, if 'number_NA', all people with more NA than cutoff are deleted
    ret_transf:             Boolean to specify whether the transformed dataframe should be returned or not
    returns:
        adjacency 
        node list
        """
    # Cast cutoff to integer if necessary
    if not isinstance(cutoff, int):
        cutoff = int(cutoff)
    
    if filter_method not in ['number_NA','number_nodes']:
        print("Unknown filter method " + filter_method + " number_NA is used")
        filter_method = 'number_NA'
    
    # Start and end date parsing
    if start_date is not None:
        try:
            start_datetime = datetime.strptime(start_date, '%d_%m_%Y')
        except ValueError:
            print("Invalid start time format. Must be string with format dd_mm_yyyy")
    else:
        start_datetime = datetime.strptime('01/01/2000', '%d/%m/%Y')
    
    if end_date is not None:    
        try:
            end_datetime = datetime.strptime(end_date, '%d_%m_%Y')
        except ValueError:
            print("Invalid end time format. Must be string with format dd_mm_yyyy")
    else:
        end_datetime = datetime.strptime('01/01/2019', '%d/%m/%Y')
        
    # Load data from datapath
    data = pd.read_csv(datapath, sep=',',lineterminator='\n', encoding='utf-8',
                       engine='c', low_memory=False) 
    
    
    # Need to keep: 
    keep_columns = ['AffairShortId','AffairTitle','VoteDate','CouncillorId','CouncillorName',
                    'CouncillorYes','CouncillorNo','CouncillorAbstain',
                    'CouncillorNotParticipated', 'CouncillorExcused','CouncillorPresident\r']
    
    data = data[keep_columns]
    data = data.rename(columns={'CouncillorPresident\r':'CouncillorPresident'})
    data['VoteDate'] = data['VoteDate'].apply(lambda x: datetime.strptime(x[4:15],
                                                                            '%b %d %Y'))
    
    # Filter by date
    data = data[((data.VoteDate >= start_datetime) & (data.VoteDate <= end_datetime))]
    # Delete all votes concerning "Ordnungsanträge"
    data = data[~((data.AffairShortId == 1) | (data.AffairShortId == 2))]
    # Create list of nodes containing name and Id number of councillor
    nodes = data[['CouncillorId','CouncillorName']].drop_duplicates(keep='first')
    # List of all affairs with their Id number. The votes of one affair corresponds to one feature of the nodes
    affairs = data[['AffairShortId','AffairTitle']].drop_duplicates(keep='first')
    # A feature of a node is equal to the vote concerning a certain affair (-> one affair_id represents one feature)
    # Replace the Affair Ids by a new index (feature index)
    affairid2feature = affairs[['AffairShortId']]
    affairid2feature.insert(1,'feature_idx',np.arange(1,affairid2feature.shape[0]+1))
    affairid2feature = affairid2feature.set_index('AffairShortId')
    
    # Convert 'Yes','No','Abstain' etc. to a numerical value
    data_with_num_val = assign_val_to_vote(data)
    # Replace each affair with a feature index instead of the affairShortId
    data_with_num_val = data_with_num_val[['CouncillorId','AffairShortId', 'value']]
    data_with_num_val = data_with_num_val.join(affairid2feature, on='AffairShortId')
    data_with_num_val = data_with_num_val.drop(columns='AffairShortId')
    data_with_num_val = data_with_num_val.set_index('feature_idx')
    
    # Dataframes containing complete feature index
    complete_feat_idx = affairid2feature[['feature_idx']].set_index('feature_idx')   
    # Start the transformed dataset with just an index column
    data_transformed = complete_feat_idx.copy()
    
     # Make dataframe from feature vectors with councillorId as column labels
    for councId in nodes.loc[:,'CouncillorId']:
        feature_vec = extract_feature_vec(data_with_num_val, councId, complete_feat_idx)
        data_transformed = pd.concat([data_transformed, feature_vec], axis=1)
    
    # Make councillorID row index
    data_transformed = data_transformed.T

    print("(Nbr. of councillors, nbr. of votes) before filter: {0}".format(data_transformed.shape))
    # Filter out councillors based on number of NAs
    if filter_method == 'number_NA':
        # could use DataFrame.dropna here but useful to return variable with sum per row for analysis
        nbr_na_per_row = data_transformed.isna().sum(axis=1)
        data_transformed = data_transformed[~(nbr_na_per_row > cutoff)]
    elif filter_method == 'number_nodes':
        nbr_na_per_row = data_transformed.isna().sum(axis=1)
        nrows_least_na = nbr_na_per_row.nsmallest(n=cutoff)
        data_transformed = data_transformed[data_transformed.index.isin(nrows_least_na.index)]
    else:
        raise ValueError
        ("Unsupported filter type. Unfiltered data is returned")
    
    # Features for which at least one of the remaining councillors has NA are removed
    # in order to eliminate any remaining NA in the data
    data_transformed = data_transformed.dropna(axis='columns',how='any')
    
    print("(Nbr. of councillors, nbr. of votes) after filter: {0}".format(data_transformed.shape))
    
    data_transformed.reset_index(inplace=True)
    data_transformed = data_transformed.rename(columns={'index':'Counc_Id'})
    # Create node index in order to identify councillors in the network later on
    node_index = data_transformed[['Counc_Id']]
    node_index = node_index.astype(int)
    node_index = node_index.join(nodes.set_index(['CouncillorId']), on='Counc_Id')
    node_index.index.names = ['node_idx']
    
    # Rename row and column index
    data_transformed = data_transformed.drop(columns=['Counc_Id'])
    data_transformed.index.names = ['node_id']
    data_transformed.columns.names = ['features']
    
    # Calculate adjacency matrix
    adjacency = get_adjacency(data_transformed)
    
    if ret_transf:
        return data_transformed, adjacency, node_index, nbr_na_per_row
    else:
        return adjacency, node_index, nbr_na_per_row
    
    
    
    
    
def get_adjacency(dataframe):
    """ Creates the adjacency matrix from dataframe """
    
    # Number of nodes in the graph
    n_nodes = dataframe.shape[0]

    # Calculate distances. Due to the high dimensional data (> 1300 dimensions) the cosine distance is chosen
    distances = np.zeros((n_nodes, n_nodes))
    
    for i, a in dataframe.iterrows():
        for j, b in dataframe.iterrows():
            dot_product = np.dot(a,b)
            distances[i,j] = 1 - dot_product/(np.linalg.norm(a,2)*np.linalg.norm(b,2))

    # Weights (gaussian) are assigned to each link based on the distance  
    kernel_width = distances.std()
    weights = np.exp(-distances**2 / (2*kernel_width**2))

    # Set main diagonal to zero (No self-loops)
    np.fill_diagonal(weights,0)
    adjacency = weights.copy()
    return adjacency

def assign_party_to_names(party_membership_list_path, namelist):
    """ Adds a column containing the party membership of the councillors specified in namelist
    party_membership_list_path:     Path to a file that contains a list with names and parties
                                    Example: '../data/Ratsmitglieder_1848_FR.csv'
    namelist:                       pd.DataFrame that contains a column of names of councillors 
                                    whose party association should be found
    returns:
        namelist_with_parties:      pd.DataFrame identical to namelist but with an added column 
                                    with the party associations"""
    
    if not isinstance(namelist, pd.DataFrame):
        raise TypeError("Namelist must be a pd.DataFrame")
        return None
    if not 'CouncillorName' in namelist.columns:
        raise KeyError("Namelist must contain a column labeled 'CouncillorName'")
        return None
    #List of all members with their party
    all_members_cn = pd.read_csv(party_membership_list_path, sep=';',lineterminator='\n')   
    all_members_cn = all_members_cn[['FirstName','LastName','PartyAbbreviation']]
    #Concatenate first and last name
    
    all_members_cn['FullName'] = all_members_cn['LastName'].str.cat(all_members_cn['FirstName'],sep=' ')   
    all_members_cn = all_members_cn.drop(columns=['LastName','FirstName'])
    #Remove duplicate 
    all_members_cn = all_members_cn[['FullName','PartyAbbreviation']].drop_duplicates(subset=['FullName'])
    namelist_with_parties = namelist.join(all_members_cn.set_index('FullName'), on='CouncillorName')
    
    n_no_party = len(namelist_with_parties) - namelist_with_parties['PartyAbbreviation'].count()
    
    if n_no_party != 0:
        print("{0} councillors couldn't be associated to a party".format(n_no_party))
    return namelist_with_parties

def make_spy_plot(adjacency):
    plt.spy(adjacency, markersize=0.050000)
    plt.title('Adjacency matrix', pad=15.0)
    plt.imshow(adjacency, cmap=plt.cm.Blues);
    plt.colorbar();
    plt.show()
    



    
                                    
    
