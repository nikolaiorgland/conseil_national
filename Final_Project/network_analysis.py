# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:15:35 2019

@author: silus
"""
import numpy as np
from load_and_preprocessing import get_adjacencies_per_year

def compute_community_loyality(community_labels, leg, years_of_leg):
    
    assert isinstance(community_labels, dict)
    assert isinstance(leg, list)
    assert isinstance(years_of_leg, list)
    assert len(leg) == len(years_of_leg)
    
    adjacencies = get_adjacencies_per_year(leg, years_of_leg)