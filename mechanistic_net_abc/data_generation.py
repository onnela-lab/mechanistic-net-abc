# -*- coding: utf-8 -*-
"""
Functions to generate sets of data from the model, in an ABC fashion.
"""

import pandas as pd
from joblib import Parallel, delayed
from mechanistic_net_abc.utility import simulate_weights, simulate_weights_truncated, _merge_dict, _sample_from_priors
from mechanistic_net_abc.models import mixture_model_simulation
from mechanistic_net_abc.summaries import compute_summaries_undirected, compute_many_summaries_undirected

### Functions only for contact network simulation ###

def data_indiv_simulation(G_seed, num_nodes, func_mechanisms,
                          prior_args_mechanisms = None, fixed_args_mechanisms = None,
                          min_weight = None, max_weight = None, many_summaries = False):
    """ Generate one simulated data, on which summary statistics are computed.
    
    This function generates one ABC simulation from a model of mixture
    of mechanisms, in the sense that:
        1) mechanism weights are simulated;
        2) parameter mechanisms are generated too;
        3) given steps one and two, a model of mixture of mechanisms is used
        to generate a graph with num_nodes nodes;
        4) summary statistics are computed according to the function
        compute_summaries_undirected in the module summaries.py.
    
    Args:
        G_seed (networkx.classes.graph.Graph):
            a networkx graph to update according to the mixture of mechanisms.
        num_nodes (int):
            the number of nodes in the simulated contact network.
        func_mechanisms (list of functions):
            a list of functions, where each corresponds to the application of a
            mechanism.
        prior_args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G), and each
            values is a scipy.stats frozen distribution 
            (scipy.stats._distn_infrastructure.rv_frozen),
            in this case the .rvs() attribute of this object will be used to
            simulate a parameter value according to this prior distribution.
            For example, if for the first mechanism, m has as prior a uniform 
            discrete distribution on [1,2]; and the second mechanism a uniform
            discrete distribution on [1,3], it gives:
                dist1 = scipy.stats.randint(1,3)
                dist2 = scipy.stats.randint(1,4)
                prior_args_mechanisms = [{'m':dist1}, {'m':dist2}].
        fixed_args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G) that
            do not have a prior function on them, and the values are the
            fixed parameter values.
        min_weight (list):
            each component indicates the minimal possible value for a 
            corresponding mechanism weight.
        max_weight (list):
            each component indicates the maximal possible value for a 
            corresponding mechanism weight.
        many_summaries (bool):
            if True, a very large number of summary statistics are computed,
            otherwise only 9. By default False. For experimentation purpose.
                
    Returns:
        dict_weights (dict):
            a dictionary that contains the weight values used for each
            mechanisms, in the model of mixture of mechanisms.
            The keys are labeled 'weight_mech_i' where i is the mechanism
            number.
        dict_params (dict):
            a dictionary that contains the simulated parameter values for all
            mechanisms. The keys are the parameter name + 'mech_i' where i is
            the mechanism number.
        dict_summaries (dict):
            a set of summary statistics computed on the network simulated
            from the model of mixture of mechanisms.
    
    """

    num_mech = len(func_mechanisms)
    
    if prior_args_mechanisms is None:
        prior_args_mechanisms = [dict() for idx in range(num_mech)]
    if fixed_args_mechanisms is None:
        fixed_args_mechanisms = [dict() for idx in range(num_mech)]
    
    if min_weight == None:
        min_weight = [0]*len(func_mechanisms)
    if max_weight == None:
        max_weight = [1]*len(func_mechanisms)
    
    # Generate the mechanism weights
    if (True in [min_weight[i] > 0 for i in range(num_mech)]) or (True in [max_weight[i] < 1 for i in range(num_mech)]):
        weights = simulate_weights_truncated(num_mech, min_weight, max_weight)
    else:
        weights = simulate_weights(num_mech)
    
    sim_values_args_mechanisms = _sample_from_priors(prior_args_mechanisms, num_mech)
    
    # Fuse the dictionaries of simulated and fixed parameters together
    args_mechanisms = [_merge_dict(sim_values_args_mechanisms[idx],
                                   fixed_args_mechanisms[idx]) for idx in range(num_mech)]
    
    # Simulate from the model.
    G_sim = mixture_model_simulation(G_seed, num_nodes, weights,
                                     func_mechanisms, args_mechanisms)
    
    # Compute the summary statistics on the simulated data
    if many_summaries:
        dict_summaries = compute_many_summaries_undirected(G_sim)        
    else:        
        dict_summaries = compute_summaries_undirected(G_sim)
    
    # Format the weights:
    dict_weights = {'weight_mech_'+str(idx+1): weights[idx] for idx in range(num_mech)}
        
    # Format the parameter values simulated:
    dict_params = dict()
    for idx in range(num_mech):
        for (key, toto) in sim_values_args_mechanisms[idx].items():
            dict_params[key+'_mech_'+str(idx+1)] = toto
    
    return dict_weights, dict_params, dict_summaries


def data_ref_table_simulation(G_seed, num_sim, num_nodes, func_mechanisms, 
                              prior_args_mechanisms, fixed_args_mechanisms,
                              num_cores = 1, min_weight = None, max_weight = None,
                              many_summaries = False):
    """ Generation of a reference table under a model of mixture of mechanisms.
    
    Function to generate a reference table with num_sim simulated data from
    a model of mixture of mechanisms. The specification of the model is achieved
    thanks to the arguments func_mechanisms. Parameters for each mechanisms are 
    simulated using the str expression in prior_args_mechanisms, and non random
    parameters are specified in fixed_args_mechanisms. The summary statistics 
    computed are the ones defined in the module summaries.py. The simulation can
    be performed in parallel when specifying num_cores larger than 1.

    Args:
        G_seed (networkx.classes.graph.Graph):
            a networkx graph to update according to the mixture of mechanisms.
        num_sim (int):
            the number of elements to simulate for the reference table, i.e.
            the number of simulated data.
        num_nodes (int):
            the number of nodes in the simulated network.
        func_mechanisms (list of functions):
            a list of functions, where each corresponds to the application of a
            mechanism.
        prior_args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G), and each
            values is a scipy.stats frozen distribution 
            (scipy.stats._distn_infrastructure.rv_frozen),
            in this case the .rvs() attribute of this object will be used to
            simulate a parameter value according to this prior distribution.
            For example, if for the first mechanism, m has as prior a uniform 
            discrete distribution on [1,2]; and the second mechanism a uniform
            discrete distribution on [1,3], it gives:
                dist1 = scipy.stats.randint(1,3)
                dist2 = scipy.stats.randint(1,4)
                prior_args_mechanisms = [{'m':dist1}, {'m':dist2}].
        fixed_args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G) that
            do not have a prior function on them, and the values are the
            fixed parameter values.
        num_cores (int):
            the number of CPU cores to use for the parallel generation of data.
        min_weight (list):
            each component indicates the minimal possible value for a 
            corresponding mechanism weight.
        max_weight (list):
            each component indicates the maximal possible value for a 
            corresponding mechanism weight.
        many_summaries (bool):
            if True, a very large number of summary statistics are computed,
            otherwise only 9. By default False. For experimentation purpose.
            
    Returns:
        df_weights (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            simulated mechanism weights (in columns).
        df_params (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            simulated parameter values (in columns).
        df_summaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
            
    """

    if num_cores > 1:
        simulated_data = Parallel(n_jobs = num_cores)(delayed(data_indiv_simulation)(G_seed = G_seed,
                                  num_nodes = num_nodes,
                                  func_mechanisms = func_mechanisms,
                                  prior_args_mechanisms = prior_args_mechanisms,
                                  fixed_args_mechanisms = fixed_args_mechanisms,
                                  min_weight = min_weight,
                                  max_weight = max_weight,
                                  many_summaries = many_summaries) for sim in list(range(num_sim)) )

    elif num_cores == 1:
        simulated_data = [data_indiv_simulation(G_seed = G_seed,
                                                num_nodes = num_nodes, 
                                                func_mechanisms = func_mechanisms,
                                                prior_args_mechanisms = prior_args_mechanisms,
                                                fixed_args_mechanisms = fixed_args_mechanisms,
                                                min_weight = min_weight,
                                                max_weight = max_weight,
                                                many_summaries = many_summaries) for sim in list(range(num_sim)) ]
    else:
        raise ValueError('Invalid value entered for num_cores.')

    list_weights = []
    list_params = []
    list_summaries = []
    
    num_sim = len(simulated_data)
    
    for simIdx in range(num_sim):
        list_weights += [simulated_data[simIdx][0]]
        list_params += [simulated_data[simIdx][1]]
        list_summaries += [simulated_data[simIdx][2]]
                
    df_weights = pd.DataFrame(list_weights)
    df_params = pd.DataFrame(list_params)
    df_summaries = pd.DataFrame(list_summaries)
    
    return df_weights, df_params, df_summaries