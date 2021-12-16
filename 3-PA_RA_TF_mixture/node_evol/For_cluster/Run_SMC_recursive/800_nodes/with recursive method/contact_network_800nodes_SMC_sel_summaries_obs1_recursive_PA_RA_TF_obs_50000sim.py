# -*- coding: utf-8 -*-
"""
In this example, we generate Barabasi-Albert networks, i.e. the only underlying
mechanism is the preferential attachment, and we want to identify that this
mechanism dominates thanks to a model of mixture of mechanisms.
We will include the pereferential attachment mechanism in our model of mixture 
of mechanisms, as well as other noise mechanisms whose weights should be low 
after inference.
"""

import networkx as nx
import pandas as pd
import numpy as np
import scipy.stats as ss
import time
import matplotlib.pyplot as plt
import os
import random
import pickle

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
from sklearn.neighbors import NearestNeighbors#, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

#######################################################
### Define the general setting
#######################################################

example_number = "_PA_RA_TF_obs"
dist_SMC_number = "_50000sim"
max_sim = 50000

### About observed data
num_nodes = 800     # Number of observed nodes
m_PA = 4            # True parameter for the preferential attachment
m_RA = 4            # and random attachment
m_seed = 4          # m parameter for the seed network (a fixed BA model with m_true)
num_nodes_seed = 10 # Number of nodes in the seed graph (a fixed BA model with m_seed)

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

min_weights = [0,0,0]
max_weights = [1,1,1]

# Number of simulated data in the first reference table (used for summary 
# selection, and distance parameters computation)
num_sim = 100000
num_sim_sel_train = 40000
num_sim_sel_val = 10000
num_sim_knn = 50000

# When using k-NN ABC:
num_neigh = 200

### Set the working directories
os.chdir("/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network")

dir_save_data = "/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network/generate_ref_table/data_"+str(num_nodes)+"nodes"+example_number
# Possibly create a folder to save the simulated data and results
if not os.path.exists(dir_save_data):
    os.makedirs(dir_save_data)

nbr_obs_SMC = 1
summaries_txt = "sel_summaries_recursive"
sel_or_not = "selected_summaries_recursive"

pred_type="individual"

dir_save_results = "/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network/run_SMC_ABC_algorithms/"+sel_or_not+"/results_"+str(num_nodes)+"nodes_SMC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number
if not os.path.exists(dir_save_results):
    os.makedirs(dir_save_results)

#######################################################
### Simulate some observed networks from the BA model
#######################################################

# Generate the seed network
G_seed = nx.barabasi_albert_graph(n = num_nodes_seed,
                                  m = m_seed,
                                  seed = 123)

# Observed network 1
G_1 = G_seed.copy()
np.random.seed(seed=123)
G_1 = mixture_model_simulation(G_seed=G_1, 
                               num_nodes=num_nodes,
                               weights=[2/3,1/6,1/6],
                               func_mechanisms=func_mechanisms,
                               args_mechanisms=[{'m':m_PA, 'degree_aug':1},{'m':m_RA},{}])
obs_sum_1 = compute_many_summaries_undirected(G_1)
df_obs_sum_1_init = pd.DataFrame([obs_sum_1])
print(obs_sum_1)

# Observed network 2
G_2 = G_seed.copy()
np.random.seed(seed=321)
G_2 = mixture_model_simulation(G_seed=G_2, 
                               num_nodes=num_nodes,
                               weights=[2/3,1/6,1/6],
                               func_mechanisms=func_mechanisms,
                               args_mechanisms=[{'m':m_PA, 'degree_aug':1},{'m':m_RA},{}])
obs_sum_2 = compute_many_summaries_undirected(G_2)
df_obs_sum_2_init = pd.DataFrame([obs_sum_2])
print(obs_sum_2)

# Observed network 3
G_3 = G_seed.copy()
np.random.seed(seed=111)
G_3 = mixture_model_simulation(G_seed=G_3, 
                               num_nodes=num_nodes,
                               weights=[2/3,1/6,1/6],
                               func_mechanisms=func_mechanisms,
                               args_mechanisms=[{'m':m_PA, 'degree_aug':1},{'m':m_RA},{}])
obs_sum_3 = compute_many_summaries_undirected(G_3)
df_obs_sum_3_init = pd.DataFrame([obs_sum_3])
print(obs_sum_3)


#########################################################################################
### Generate a reference table of size N (for distance computation and summary selection)
#########################################################################################

### This is run on a cluster, only the data are loaded below

# time1 = time.time()
# (df_weights_ref_table,
#   df_params_ref_table,
#   df_summaries_ref_table_init) = data_ref_table_simulation(G_seed = G_seed,
#                                                            num_sim = num_sim,
#                                                            num_nodes = num_nodes,
#                                                            func_mechanisms = func_mechanisms,
#                                                            prior_args_mechanisms = prior_args_mechanisms,
#                                                            fixed_args_mechanisms = fixed_args_mechanisms,
#                                                            num_cores = 1,
#                                                            min_weight = min_weights,
#                                                            max_weight = max_weights,
#                                                            many_summaries = True)
# time2 = time.time()
# print("Time to simulate the reference table: {} seconds.".format(time2 - time1))

# df_weights_ref_table.to_csv(dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
# df_params_ref_table.to_csv(dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
# df_summaries_ref_table_init.to_csv(dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)

df_weights_ref_table = pd.read_csv(filepath_or_buffer = dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv")
df_params_ref_table = pd.read_csv(filepath_or_buffer = dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+".csv")
df_summaries_ref_table_init = pd.read_csv(filepath_or_buffer = dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv")

df_weights_ref_table.min()
df_weights_ref_table.max()

###############################################################################
#### Drop the redundant features (i.e. no value variability or correlation 1)
###############################################################################

### We discard the summary statistics that present no variability

# For the reference table
df_summaries_ref_table_r = drop_redundant_features(df_summaries_ref_table_init)
# ['num_of_CC', 'num_nodes_LCC'] are removed between we have only one component

# For the observed summaries too
nunique = df_summaries_ref_table_init.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique==1].index
df_obs_sum_1_r = df_obs_sum_1_init.drop(cols_to_drop, axis=1)
df_obs_sum_2_r = df_obs_sum_2_init.drop(cols_to_drop, axis=1)
df_obs_sum_3_r = df_obs_sum_3_init.drop(cols_to_drop, axis=1)

### We also keep one summary among a cluster that present a correlation 1

plt.figure(figsize=(15,10))
correlations = df_summaries_ref_table_r.corr() # small approx error, corrected below
correlations[correlations>1] = 1
correlations[correlations<-1] = -1
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);

plt.figure(figsize=(12,5))
dissimilarity = 1 - abs(round(correlations,2))
Z = linkage(squareform(dissimilarity), 'complete')
dendrogram(Z, labels=df_summaries_ref_table_r.columns,
           orientation='top',
           leaf_rotation=90);

threshold = 0 # Search for correlation 1 clusters
labels_clust = fcluster(Z, threshold, criterion='distance')

### Keep one summary statistic per cluster (among pairs that have a correlation of 1)

cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(labels_clust):
    cluster_id_to_feature_ids[cluster_id].append(idx)

print(cluster_id_to_feature_ids) # We will discard 4 summary statistics, indexed 1, 8, 5 and 16

selected_features_clust = [v[0] for v in cluster_id_to_feature_ids.values()]
idx_cols_to_drop_clust = list(set(range(df_summaries_ref_table_r.shape[1])) - set(selected_features_clust))
cols_to_drop_clust = df_summaries_ref_table_r.columns[idx_cols_to_drop_clust]
print(cols_to_drop_clust)
# ['degree_mean', 'num_edges_LCC', 'avg_deg_connectivity_LCC', 'edge_connectivity_LCC']

# Reduce the reference table
df_summaries_ref_table = df_summaries_ref_table_r.drop(cols_to_drop_clust, axis=1)

# For the observed summaries too
df_obs_sum_1 = df_obs_sum_1_r.drop(cols_to_drop_clust, axis=1)
df_obs_sum_2 = df_obs_sum_2_r.drop(cols_to_drop_clust, axis=1)
df_obs_sum_3 = df_obs_sum_3_r.drop(cols_to_drop_clust, axis=1)

### Save the name of the summaries
name_summaries_unsel = np.array(list(df_summaries_ref_table.columns))

###############################################################################
### With this reduced number of summary statistics, I would like to highlight
### the correlation between summaries based on clusters, for RF importance graphs
###############################################################################

### We also keep one summary among a cluster that present a correlation 1

plt.figure(figsize=(15,10))
correlations_2 = df_summaries_ref_table.corr() # small approx error, corrected below
correlations_2[correlations_2>1] = 1
correlations_2[correlations_2<-1] = -1
sns.heatmap(round(correlations_2,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);

plt.figure(figsize=(12,5))
dissimilarity_2 = 1 - abs(round(correlations_2,2))
Z_2 = linkage(squareform(dissimilarity_2), 'complete')
dendrogram(Z_2, labels=df_summaries_ref_table.columns,
           orientation='top',
           leaf_rotation=90);

threshold_2 = 0.10 # Search for correlation 1 clusters
labels_clust_2 = fcluster(Z_2, threshold_2, criterion='distance')

### Keep one summary statistic per cluster (among pairs that have a correlation of 1)

cluster_id_to_feature_ids_2 = defaultdict(list)
for idx, cluster_id in enumerate(labels_clust_2):
    cluster_id_to_feature_ids_2[cluster_id].append(idx)

print(cluster_id_to_feature_ids_2)
# Assign a color to head summary based on its cluster appartenance

color_cluster_values = np.zeros(len(name_summaries_unsel))
for keys in cluster_id_to_feature_ids_2.keys():
    color_cluster_values[cluster_id_to_feature_ids_2[keys]]=keys
    
import matplotlib.colors

cmap = plt.cm.hsv
norm = matplotlib.colors.Normalize(vmin=np.min(color_cluster_values),
                                   vmax=np.max(color_cluster_values))
    

###############################################################################
### Scale the data
###############################################################################

scaler = StandardScaler()
df_summaries_ref_table_scaled = pd.DataFrame(scaler.fit_transform(df_summaries_ref_table),
                                             columns = name_summaries_unsel)
df_obs_sum_1_scaled = pd.DataFrame(scaler.transform(df_obs_sum_1),
                                   columns = name_summaries_unsel)
df_obs_sum_2_scaled = pd.DataFrame(scaler.transform(df_obs_sum_2),
                                   columns = name_summaries_unsel)
df_obs_sum_3_scaled = pd.DataFrame(scaler.transform(df_obs_sum_3),
                                   columns = name_summaries_unsel)

###############################################################################
### Create the joint respones and scale them if necessary
###############################################################################

# Drop response that are fixed (no prior on the parameter)
df_weights_params_ref_table = pd.concat([df_weights_ref_table,
                                         df_params_ref_table],
                                        axis=1, join="inner")

df_weights_params_ref_table = drop_redundant_features(df_weights_params_ref_table)

# Compute the prior standard deviation of the parameters (including weights)
std_prior_params = np.sqrt(ss.dirichlet([1]*num_mechs).var().tolist() + [prior_m_pref_att.var()] + [prior_m_rand_att.var()])

df_weights_params_scaled = df_weights_params_ref_table/std_prior_params

###############################################################################
### Split the reference table: one for summary selection (train + val), one for ABC inference
###############################################################################

# For summary statistic selection (split in two, for selection and validation)
idx_data_sel_train = range(num_sim_sel_train)
df_weights_sel_train = df_weights_ref_table.iloc[idx_data_sel_train]
df_params_sel_train = df_params_ref_table.iloc[idx_data_sel_train]
df_summaries_scaled_sel_train = df_summaries_ref_table_scaled.iloc[idx_data_sel_train]
# df_summaries_noised_scaled_sel = df_summaries_ref_table_noised_scaled[idx_data_sel]
covariates_all_scaled_sel_train = df_summaries_scaled_sel_train.to_numpy(copy=True)

idx_data_sel_val = range(num_sim_sel_train, num_sim_sel_train+num_sim_sel_val)
df_weights_sel_val = df_weights_ref_table.iloc[idx_data_sel_val]
df_params_sel_val = df_params_ref_table.iloc[idx_data_sel_val]
df_summaries_scaled_sel_val = df_summaries_ref_table_scaled.iloc[idx_data_sel_val]
# df_summaries_noised_scaled_val = df_summaries_ref_table_noised_scaled[idx_data_sel_val]
covariates_all_scaled_sel_val = df_summaries_scaled_sel_val.to_numpy(copy=True)

# For k-NN ABC (only the summaries are needed in the end)
idx_data_knn = range(num_sim_sel_train+num_sim_sel_val, num_sim)
df_weights_knn = df_weights_ref_table.iloc[idx_data_knn]
df_params_knn = df_params_ref_table.iloc[idx_data_knn]
df_summaries_scaled_knn = df_summaries_ref_table_scaled.iloc[idx_data_knn]
# df_summaries_noised_scaled_knn = df_summaries_ref_table_noised_scaled[idx_data_knn]
covariates_all_scaled_knn = df_summaries_scaled_knn.to_numpy(copy=True)

# covariates_all_noised_scaled_sel = df_summaries_noised_scaled_sel_train.to_numpy(copy=True)
# num_summaries_noised = df_summaries_ref_table_noised_scaled.shape[1]

### Create the response (multidimensional)
# Unscaled
df_weights_params_sel_train = df_weights_params_ref_table.iloc[idx_data_sel_train]
response_all_sel_train = df_weights_params_sel_train.to_numpy(copy=True)

df_weights_params_sel_val =  df_weights_params_ref_table.iloc[idx_data_sel_val]
response_all_sel_val = df_weights_params_sel_val.to_numpy(copy=True)

df_weights_params_knn =  df_weights_params_ref_table.iloc[idx_data_knn]
response_all_knn = df_weights_params_knn.to_numpy(copy=True)

# Scaled
df_weights_params_scaled_sel_train = df_weights_params_scaled.iloc[idx_data_sel_train]
response_all_scaled_sel_train = df_weights_params_scaled_sel_train.to_numpy(copy=True)

df_weights_params_scaled_sel_val =  df_weights_params_scaled.iloc[idx_data_sel_val]
response_all_scaled_sel_val = df_weights_params_scaled_sel_val.to_numpy(copy=True)

df_weights_params_scaled_knn =  df_weights_params_scaled.iloc[idx_data_knn]
response_all_scaled_knn = df_weights_params_scaled_knn.to_numpy(copy=True)


########################### RF selection ######################################

###############################################################################
### Run the SMC-ABC algorithm on the summary statistics selected with RF
###############################################################################

os.chdir("/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network/recursive_and_nonRecursive_selection/saved_rankings_200nodes_PA_RA_TF_obs")
recursive_selected_summaries_obs = pickle.load( open( "recursive_selected_summaries_obs"+str(nbr_obs_SMC)+"_"+pred_type+example_number+".p", "rb" ) )

# # To choose the distance we want to reach we will use

min_weight = min_weights
max_weight = max_weights
alpha = 0.1
scale_factor = 2
weight_perturbation="Gaussian"
num_acc_sim = 200
distance_func = distance_euclidean_std
idx_selection_from_rf = recursive_selected_summaries_obs

# To choose the distance we want to reach we will use

std_values_sel = pd.DataFrame( [df_summaries_ref_table.iloc[:,idx_selection_from_rf].apply(np.std)] )
distance_args_sel = {'std_values':std_values_sel}

# Here are the ABC-RSMC parameters that changed
distance_args = distance_args_sel
sel_sum_names = list(name_summaries_unsel[idx_selection_from_rf]) # We do use selected summaries here

###############################################################################
###### For the first observed data ######

df_obs_sum_RABC = df_obs_sum_1[sel_sum_names]

# To determine the threshold final, let's look at the distance obtained
# on the best data from the ABC simulated reference table.
vec_distances_with_obs_sel = np.array([distance_euclidean_std(df_sim_summaries = df_summaries_ref_table[sel_sum_names].iloc[[i]],
                                                                df_obs_summaries = df_obs_sum_RABC,
                                                                std_values = distance_args_sel['std_values'])
                                        for i in idx_data_knn])
threshold_final = 0

threshold_init = 50 # A large value at least

time1 = time.time()
(df_weights_RABC_obs_selected,
  df_params_RABC_obs_selected,
  df_dist_acc_RABC_obs_selected,
  sim_count_total_obs_selected,
  threshold_values_obs_selected) = abc_RSMCABC(G_seed=G_seed,
                                                num_nodes=num_nodes,
                                                func_mechanisms=func_mechanisms,
                                                prior_args_mechanisms=prior_args_mechanisms,
                                                fixed_args_mechanisms=fixed_args_mechanisms,
                                                min_weight=min_weight,
                                                max_weight=max_weight,
                                                threshold_init=threshold_init,
                                                threshold_final=threshold_final,
                                                alpha=alpha,
                                                scale_factor=scale_factor,
                                                weight_perturbation=weight_perturbation,
                                                num_acc_sim=num_acc_sim,
                                                df_observed_summaries=df_obs_sum_RABC,
                                                distance_func=distance_func,
                                                distance_args=distance_args,
                                                sel_sum_names=sel_sum_names,
                                                max_sim = max_sim)
time2 = time.time()
print("Time ABC-RSMC (with selected summaries): ", time2 - time1)

print("Total number of simulations: ", sim_count_total_obs_selected)

df_weights_RABC_obs_selected.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number+".csv", index=False)
df_params_RABC_obs_selected.to_csv(dir_save_results+"/df_params_RABC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number+".csv", index=False)
df_dist_acc_RABC_obs_selected.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number+".csv", index=False)

df_weights_RABC_obs_selected = pd.read_csv(filepath_or_buffer = dir_save_results+"/df_weights_RABC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number+".csv")
df_params_RABC_obs_selected = pd.read_csv(filepath_or_buffer = dir_save_results+"/df_params_RABC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number+".csv")
df_dist_acc_RABC_obs_selected = pd.read_csv(filepath_or_buffer = dir_save_results+"/df_dist_acc_RABC_obs"+str(nbr_obs_SMC)+"_"+summaries_txt+example_number+dist_SMC_number+".csv")

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs_selected.apply(np.mean)))
print(np.array(df_weights_RABC_obs_selected.apply(np.median)))

print(df_weights_RABC_obs_selected.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs_selected.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(df_weights_RABC_obs_selected.iloc[:,0], alpha=0.4, label='pref. att. weight')
plt.hist(df_weights_RABC_obs_selected.iloc[:,1], alpha=0.4, label='rand. att. weight')
plt.legend()
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs_selected.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs_selected.iloc[:,0], alpha=0.4, label='pref. att. m param.', bins=range(1,11))
plt.hist(df_params_RABC_obs_selected.iloc[:,1], alpha=0.4, label='rand. att. m param.', bins=range(1,11))
plt.legend()
plt.show()