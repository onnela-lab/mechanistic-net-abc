# -*- coding: utf-8 -*-
"""
Implementation to simulate from a model of mixture of mechanisms for network
formation.
"""

import inspect
import numpy as np

def mixture_model_simulation(G_seed, num_nodes, weights, func_mechanisms, args_mechanisms):
    """ Simulate a network according to a model of mixture of mechanisms.
    
    This function simulates a network starting from G_seed, until num_nodes are
    added to the simulated data, using a model of mixture of mechanisms.
    The user gives the mechanism weights, the list of mechanisms (as functions),
    and their arguments values (as list of dictionaries).
        
    Args:
        G_seed (networkx.classes.graph.Graph):
            a networkx graph to update according to the mixture of mechanisms.
        num_nodes (int):
            the number of nodes in the simulated network.
        weights (list of float):
            the list of float numbers that correspond to the mixture weights.
        func_mechanisms (list of functions):
            a list of functions, where each corresponds to the application of a
            mechanism.
        args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G), and the
            values are the parameter values to use in the simulation step 
            from the model of mixture.
            
    Returns:
        G_sim (networkx.classes.graph.Graph):
            a simulated networkx graph with num_nodes nodes.
                
    """

    ### Preliminary checks
    # Check that G_seed has well less than num_nodes
    if G_seed.number_of_nodes() >= num_nodes:
        raise ValueError('num_nodes must be larger than the number of nodes in G_seed.')   

    # Check that weights and func_mech and args_mech have the same length
    if len(weights) != len(func_mechanisms) or len(weights) != len(args_mechanisms):
        raise ValueError('The length of weights, func_mechanisms and args_mechanisms do not match.')

    # Check that G is not an argument in one of the dict in args_mechanisms,
    # and that the arguments specifed do exist for the mechanism function
    for idx_mech, dict_args in enumerate(args_mechanisms):
        if "G" in dict_args:
            raise ValueError('The graph to modify (G_seed) must not need to be included in args_mechanisms.')
        if not all(arg in inspect.getfullargspec(func_mechanisms[idx_mech])[0] for arg in dict_args):
            raise ValueError('An argument specified in args_mechanisms does not exist in the corresponding mechanism definition.')
    
    ### Core of the function
    # Copy the seed network to avoid its modification 
    G_sim = G_seed.copy()
    
    # Simulate the network until having num_nodes nodes in it
    while G_sim.number_of_nodes() < num_nodes:
        _mixture_mechanisms_step(G_sim, weights, func_mechanisms, args_mechanisms)
    
    return G_sim
    
    
def _mixture_mechanisms_step(G, weights, func_mechanisms, args_mechanisms, seed = None):
    """ Simulate one step of a model of mixture of mechanisms for network formation.
    
    This function simulates one step of a mixture of mechanisms.
    The use gives the mechanism weights, the list of mechanisms (as functions),
    and their arguments values (as list of dictionaries).
        
    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph to update according to the mixture of mechanism.
        weights (list of float):
            the list of float numbers that correspond to the mixture weights.
        func_mechanisms (list of functions):
            a list of functions, where each correspond to the application of a
            mechanism.
        args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G), and the
            values are the parameter values to use in the simulation step 
            from the model of mixture.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for mechanism sampling. None by default.    
            
    Returns:
        None:
            this function directly modify the existing graph G.
                
    """
    
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
    
    # Sample from a multinonial distribution to determine which mechanism will
    # be used
    sampled_mn = random_seed.multinomial(1, weights)
    idx_mechanism = np.where(sampled_mn == 1)[0][0]
    
    # Use the mechanism with the specified parameters other than G
    func_mechanisms[idx_mechanism](G, **args_mechanisms[idx_mechanism])