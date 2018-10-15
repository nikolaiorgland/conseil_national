import pandas as pd
import numpy as np

# This function assigns a value to each affair depending on how the councillor voted.
# Yes is 1, no is 0, abstain or absence is 0.5

def assign_val_to_vote(data):
    counc_votes = data.copy()
    nbr_votes = counc_votes.shape[0]
    counc_votes.insert(len(counc_votes.columns),'value',np.zeros(nbr_votes))
    counc_votes.loc[counc_votes.CouncillorYes == '1','value'] = 1
    counc_votes.loc[counc_votes.CouncillorNo == '1','value'] = 0
    counc_votes.loc[(counc_votes.CouncillorAbstain == '1') | 
                        (counc_votes.CouncillorExcused == 1) | 
                        (counc_votes.CouncillorNotParticipated == 1),'value'] = 0.5
    
    return counc_votes
    #return counc_votes[['AffairShortId','value']]

def extract_feature_vec(val_votes, councId, complete_feat_idx):
    # Look for the votes of one particular councillor and keep values and affair Ids
    feature_vec = val_votes.loc[val_votes.CouncillorId == councId, :]
    feature_vec = feature_vec.drop(columns=['CouncillorId'])
    
    # For multiple votes with the same Id, only keep the deciding vote (the first one in the df)
    feature_vec = feature_vec[~feature_vec.index.duplicated(keep='first')]
    
    # Councillors that have missing data for an affair get assigned NA in the feature vector
    # (Which can be the case for people who resign for example)
    feature_vec = complete_feat_idx.join(feature_vec)
    
    # Rename the value column to have the councillor Id as name for identification later on
    feature_vec = feature_vec.rename(columns={'value':str(councId)})
    return feature_vec



def downcast_column(df, column, dtype):
    return pd.to_numeric(df[column], downcast=dtype)


    
    