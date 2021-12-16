# -*- coding: utf-8 -*-
"""
@author: Louis
Code to concatenate smaller reference tables simulated separately.
"""

import pandas as pd
import numpy as np

import os




#######################################################
### Define the general setting
#######################################################

example_number = "_PA_RA_TF_obs"

list_num_nodes = [50,100,400,800]

for num_nodes in list_num_nodes:
    
    ### Set the working directories
    os.chdir("/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network/generate_ref_table")
    # os.chdir("E:/postdoc/hiv_mechanistic_network/mechanistic_network_models/2-PA_RA_TF_contact_network")
    dir_save_data = "data_"+str(num_nodes)+"nodes"+example_number
    
    # Possibly create a folder to save the simulated data and results
    if not os.path.exists(dir_save_data):
        os.makedirs(dir_save_data)
        
    num_rep = 40
    num_sim_per_rep = 2500
    
    num_sim = num_rep*num_sim_per_rep
    
    list_df_weights = []
    list_df_params = []
    list_df_summaries = []
    
    for rep_idx in np.arange(1,num_rep+1,1):
        list_df_weights += [pd.read_csv(filepath_or_buffer = dir_save_data+"/df_weights_ref_table_size_"+str(num_sim_per_rep)+example_number+"_rep"+str(rep_idx)+".csv")]
        list_df_params += [pd.read_csv(filepath_or_buffer = dir_save_data+"/df_params_ref_table_size_"+str(num_sim_per_rep)+example_number+"_rep"+str(rep_idx)+".csv")]
        list_df_summaries += [pd.read_csv(filepath_or_buffer = dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim_per_rep)+example_number+"_rep"+str(rep_idx)+".csv")]
    
    df_weights_ref_table = pd.concat(list_df_weights)
    df_params_ref_table = pd.concat(list_df_params)
    df_summaries_ref_table_init = pd.concat(list_df_summaries)
    
    df_weights_ref_table.to_csv(dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
    df_params_ref_table.to_csv(dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
    df_summaries_ref_table_init.to_csv(dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)

