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

def join_votes_to_affairs(affair_idx, val_votes, councId):
    val_votes = val_votes.set_index('AffairShortId')
    affair_vote_idx = affair_idx.join(val_votes, on='AffairShortId')
    affair_vote_idx = affair_vote_idx.rename(columns={'value':councId})
    return affair_vote_idx
    
    