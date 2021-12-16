import networkx as nx
import pandas as pd
import numpy as np
import scipy.stats as ss
import time
import matplotlib.pyplot as plt
import os
import random
import math
import seaborn as sns
# import shap

from collections import Counter
from collections import defaultdict

from mechanistic_network_models.summaries import compute_many_summaries_undirected
from mechanistic_network_models.mechanisms import preferential_attachment_growth, random_attachment_growth, triangle_formation_node_addition
from mechanistic_network_models.data_generation import data_ref_table_simulation
from mechanistic_network_models.data_generation import data_indiv_simulation
from mechanistic_network_models.utility import drop_redundant_features
from mechanistic_network_models.abc import abc_RSMCABC, distance_euclidean_std
from mechanistic_network_models.models import mixture_model_simulation

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import sys

#######################################################
### Define the general setting
#######################################################

example_number = "_PA_RA_TF_obs"

rep_idx = int(sys.argv[1])
print(rep_idx)

### About observed data
num_nodes = 400     # Number of observed nodes
m_true = 4          # True parameter for the preferential attachment
num_nodes_seed = 10 # Number of nodes in the seed graph (a fixed BA model with m_true)

### About the model of mixture
mech1 = preferential_attachment_growth
mech2 = random_attachment_growth
mech3 = triangle_formation_node_addition
func_mechanisms = [mech1, mech2, mech3]
num_mechs = len(func_mechanisms)
### About the ABC setting

# Practical prior upper bound:
# Define the upper bound for priors on m parameters as the number of nodes in the seed network,
# otherwise it is not possible to use a larger value at the start of the simulation process.
max_m_value = num_nodes_seed

prior_m_pref_att = ss.randint(1, max_m_value+1)
prior_m_rand_att = ss.randint(1, max_m_value+1)
prior_args_mechanisms = [{'m':prior_m_pref_att}, {'m':prior_m_rand_att}, {}]
fixed_args_mechanisms = [{'degree_aug':1}, {}, {}]
# num_mech_params = sum([len(prior_args_mechanisms[i]) for i in range(len(prior_args_mechanisms))])

min_weights = [0,0,0]
max_weights = [1,1,1]

# Compute the prior standard deviation of the parameters (including weights)
std_prior_params = np.sqrt(ss.dirichlet([1]*num_mechs).var().tolist() + [prior_m_pref_att.var()] + [prior_m_rand_att.var()])

# Number of simulated data in the first reference table (used for summary 
# selection, and distance parameters computation)
num_sim = 2500
num_sim_sel_train = 40000
num_sim_sel_val = 10000
num_sim_knn = 50000

# When using k-NN ABC:
num_neigh = 200

### Set the working directories
os.chdir("/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network/generate_ref_table")
dir_save_data = "data_"+str(num_nodes)+"nodes"+example_number

# Possibly create a folder to save the simulated data and results
if not os.path.exists(dir_save_data):
    os.makedirs(dir_save_data)    

#######################################################
### Simulate some observed networks from the BA model
#######################################################

# Generate the seed network
G_seed = nx.barabasi_albert_graph(n = num_nodes_seed,
                                  m = m_true,
                                  seed = 123)

# Observed network 1
G_1 = G_seed.copy()
np.random.seed(seed=123)
while G_1.number_of_nodes() < num_nodes:
    preferential_attachment_growth(G = G_1, m = m_true,
                                   degree_aug = 1)
obs_sum_1 = compute_many_summaries_undirected(G_1)
df_obs_sum_1_init = pd.DataFrame([obs_sum_1])
print(obs_sum_1)

# Observed network 2
G_2 = G_seed.copy()
np.random.seed(321)
while G_2.number_of_nodes() < num_nodes:
    preferential_attachment_growth(G = G_2, m = m_true,
                                   degree_aug = 1)
obs_sum_2 = compute_many_summaries_undirected(G_2)
df_obs_sum_2_init = pd.DataFrame([obs_sum_2])
print(obs_sum_2)

# Observed network 3
G_3 = G_seed.copy()
np.random.seed(111)
while G_3.number_of_nodes() < num_nodes:
    preferential_attachment_growth(G = G_3, m = m_true,
                                   degree_aug = 1)
obs_sum_3 = compute_many_summaries_undirected(G_3)
df_obs_sum_3_init = pd.DataFrame([obs_sum_3])
print(obs_sum_3)


#########################################################################################
### Generate a reference table of size N (for distance computation and summary selection)
#########################################################################################

### This is run on a cluster, only the data are loaded below

np.random.seed(rep_idx)

time1 = time.time()
(df_weights_ref_table,
  df_params_ref_table,
  df_summaries_ref_table_init) = data_ref_table_simulation(G_seed = G_seed,
                                                           num_sim = num_sim,
                                                           num_nodes = num_nodes,
                                                           func_mechanisms = func_mechanisms,
                                                           prior_args_mechanisms = prior_args_mechanisms,
                                                           fixed_args_mechanisms = fixed_args_mechanisms,
                                                           num_cores = 1,
                                                           min_weight = min_weights,
                                                           max_weight = max_weights,
                                                           many_summaries = True)
time2 = time.time()
print("Time to simulate the reference table: {} seconds.".format(time2 - time1))

df_weights_ref_table.to_csv(dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+"_rep"+str(rep_idx)+".csv", index=False)
df_params_ref_table.to_csv(dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+"_rep"+str(rep_idx)+".csv", index=False)
df_summaries_ref_table_init.to_csv(dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+"_rep"+str(rep_idx)+".csv", index=False)

df_weights_ref_table = pd.read_csv(filepath_or_buffer = dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+"_rep"+str(rep_idx)+".csv")
df_params_ref_table = pd.read_csv(filepath_or_buffer = dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+"_rep"+str(rep_idx)+".csv")
df_summaries_ref_table_init = pd.read_csv(filepath_or_buffer = dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+"_rep"+str(rep_idx)+".csv")