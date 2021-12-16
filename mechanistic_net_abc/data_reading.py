# -*- coding: utf-8 -*-
"""
Functions for reading data.
"""

import networkx as nx
import pandas as pd

def read_household_data_filepath(filepath):
    """ Read the household data from its file path """
    
    vaccine_node_list = pd.read_table(filepath_or_buffer = filepath,
                                      header = None, sep=" ")
    
    vaccine_network = nx.Graph()
    
    for row_idx, row in vaccine_node_list.iterrows():
        vaccine_network.add_edges_from([tuple(row)])
           
    vaccine_network.remove_edges_from(nx.selfloop_edges(vaccine_network))

    return vaccine_network