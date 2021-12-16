# -*- coding: utf-8 -*-
"""
Utility functions
"""

import numpy as np
import scipy.stats as ss
import pandas as pd
from scipy.stats import rv_discrete, rv_continuous
import time

def simulate_weights(num_mech):
    """ This function simulates num_mech weights for the mechanisms.
    
    Args:
        num_mech (int):
            the number of weights to simulate, equal to the number of mechanisms.
            
    Returns:
        weights (numpy.array):
            a numpy.array that contains the simulated weights.
    
    """
    
    weights = ss.dirichlet.rvs([1]*num_mech)[0]
    return weights
  
    
def _find_paths(G, node_id, n, excludeSet = None):
    """ 
    Function to determine the list of shortest path with length n (n-edge paths),
    starting from node node_id
    """
    if excludeSet == None:
        excludeSet = set([node_id])
    else:
        excludeSet.add(node_id)
    if n==0:
        return [[node_id]]
    paths = [[node_id]+path for neighbor in G.neighbors(node_id) if neighbor not in excludeSet for path in _find_paths(G, neighbor, n-1, excludeSet)]
    excludeSet.remove(node_id)
    return paths

def _merge_dict(dict1, dict2):
    """
    Function to merge two dictionaries
    """
    res = {**dict1, **dict2}
    return res

def simulate_weights_truncated(num_mech, min_weight, max_weight):
    """ This function is a truncated version of simulate_weights.
    
    The truncation will be applied based on each component of the min_weight and
    max_weight lists.
    
    Args:
        num_mech (int):
            the number of weights to simulate, equal to the number of mechanisms.
        min_weight (list):
            each component indicates the minimal possible value for a 
            corresponding mechanism weight.
        max_weight (float):
            each component indicates the maximal possible value for a 
            corresponding mechanism weight.
            
    Returns:
        weights (numpy.array):
            a numpy.array that contains the simulated weights.
    
    """
        
    weights = ss.dirichlet.rvs([1]*num_mech)[0]
    sum_cond = np.sum([weights[i]<max_weight[i] and weights[i]>min_weight[i] for i in range(num_mech)])
    
    while sum_cond!=num_mech:
        weights = ss.dirichlet.rvs([1]*num_mech)[0]
        sum_cond = np.sum([weights[i]<max_weight[i] and weights[i]>min_weight[i] for i in range(num_mech)])
        
    return weights

def simulate_weights_truncated_old(num_mech, min_weight, max_weight):
    """ This function is a truncated version of simulate_weights.
    
    The truncation will be applied to each weight component.
    
    Args:
        num_mech (int):
            the number of weights to simulate, equal to the number of mechanisms.
        min_weight (float):
            the minimal possible value for a mechanism weight.
        max_weight (float):
            the maximal possible value for a mechanism weight.
            
    Returns:
        weights (numpy.array):
            the numpy.array that contains the simulated weights.
    
    """
        
    weights = ss.dirichlet.rvs([1]*num_mech)[0]
    
    while any(weights < min_weight) or any(weights > max_weight):
        weights = ss.dirichlet.rvs([1]*num_mech)[0]

    return weights


def drop_redundant_features(df_summaries):
    """ Function to drop the redundant features of the reference table.
    
    This function drops the features that contains the same value no matter the
    simulated data.

    Args:
        df_summaries (pandas.core.frame.DataFrame):
            the pandas DataFrame containing, for each simulated data (in rows),
            the summary statistic values (in columns).
            
    Returns:
        df_summaries_red (pandas.core.frame.DataFrame):
            df_summaries for which redundant features have been removed.
            
    """

    nunique = df_summaries.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique==1].index
    
    # Drop the redundant columns
    df_summaries_red = df_summaries.drop(cols_to_drop, axis=1)
    
    print("Columns dropped:", list(cols_to_drop))
    
    return df_summaries_red


def _is_discrete(dist):
    "To identify whether the ss.rv_frozen dist is distrete"

    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_discrete)
    else: 
        return isinstance(dist, rv_discrete)

def _is_continuous(dist):
    "To identify whether the ss.rv_frozen dist is continuous"
    
    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_continuous)
    else:
        return isinstance(dist, rv_continuous)

def _perturb_discrete_param_on_support(prior_disc, perturb_kernel):
    """ Perturb a discrete parameter thanks to a truncated Gaussian 
    distribution, and rounding of the perturbed value
    
    We use a perturbation distribution (perturb_kernel) centered at the value 
    to perturb (which is discrete). The perturbed parameter value is then 
    rounded to fall in beans of size 1, centered at the integer value. If the 
    perturbed parameter value falls in the support of the discrete prior 
    distribution for the parameter then we accept this value, otherwise we keep
    simulating to fall in the support.
    
    """
    
    # Generate the perturbed value
    perturbed_float = perturb_kernel.rvs()
    perturbed_int = np.round(perturbed_float)
    while perturbed_int < prior_disc.support()[0] or perturbed_int > prior_disc.support()[1]:
        perturbed_float = perturb_kernel.rvs()
        perturbed_int = np.round(perturbed_float)        

    return perturbed_int

def _perturb_continuous_param_on_support(prior_cont, perturb_kernel):
    """ Perturb a continuous parameter thanks to a truncated Gaussian distribution """
    
    # Generate the perturbed value
    perturbed_float = perturb_kernel.rvs()
    while perturbed_float < prior_cont.support()[0] or perturbed_float > prior_cont.support()[1]:
        perturbed_float = perturb_kernel.rvs()

    return perturbed_float

def _sample_from_priors(prior_args_mechanisms, num_mech):
    """ Function to sample from the dictionary of priors """
    
    # To store the simulated parameter values
    sim_values_args_mechanisms = [dict() for idx in range(num_mech)]
    
    # Generate parameter values from the priors, 
    # for each mechanism
    for idx in range(num_mech):
        # For each key, we simulate from the prior
        for (key, value) in prior_args_mechanisms[idx].items():
            if isinstance(value, ss._distn_infrastructure.rv_frozen):
                sim_values_args_mechanisms[idx][key] = value.rvs(random_state=ss.randint(0,4294967296).rvs())
            else:
                raise ValueError('Invalid value for the prior simulation object.')
                
    return sim_values_args_mechanisms


def _sample_from_priors_2(prior_args_mechanisms, num_mech):
    """ Function to sample from the dictionary of priors """
    
    # To store the simulated parameter values
    sim_values_args_mechanisms = [dict() for idx in range(num_mech)]
    
    # Generate parameter values from the priors, 
    # for each mechanism
    for idx in range(num_mech):
        # For each key, we simulate from the prior
        for key in prior_args_mechanisms[idx].keys():
            if isinstance(prior_args_mechanisms[idx][key], ss._distn_infrastructure.rv_frozen):
                prior_dist = prior_args_mechanisms[idx][key]
                sim_values_args_mechanisms[idx][key] = prior_dist.rvs(random_state=ss.randint(0,4294967296).rvs())
            else:
                raise ValueError('Invalid value for the prior simulation object.')
                
    return sim_values_args_mechanisms


def _current_milli_time():
    return round(time.time() * 1000)