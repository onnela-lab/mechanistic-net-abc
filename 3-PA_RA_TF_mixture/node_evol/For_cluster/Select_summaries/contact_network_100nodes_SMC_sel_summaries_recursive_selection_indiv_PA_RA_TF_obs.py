# -*- coding: utf-8 -*-
"""
@author: Louis

This example is similar to the previous example (a-___.py), but we here
add (to preferential attachment) two noise mechanisms: 
random attachment and triangle formation with node addition.
"""

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
from mechanistic_network_models.summary_selection import recursiveElimination_RFMDA_select_summaries, RFMDA_select_summaries

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

#######################################################
### Define the general setting
#######################################################

example_number = "_PA_RA_TF_obs"
max_sim = 50000

### About observed data
num_nodes = 100     # Number of observed nodes
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

dir_save_results = "/n/holyscratch01/onnela_lab/Users/lraynal/mixture_mechanisms/mechanistic_network_models/2-PA_RA_TF_obs_contact_network/recursive_and_nonRecursive_selection/saved_rankings_"+str(num_nodes)+"nodes"+example_number
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

threshold_2 = 0.50 # Search for correlation 1 clusters
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


###############################################################################
### Data visualization through PCA and PLS components
###############################################################################

### PCA analyses
pca_model = PCA(n_components = 8) # we want to project on the two first components
df_summaries_PCA = pca_model.fit_transform(df_summaries_scaled_sel_train)
df_obs_sum_1_PCA = pca_model.transform(df_obs_sum_1_scaled)
df_obs_sum_2_PCA = pca_model.transform(df_obs_sum_2_scaled)
df_obs_sum_3_PCA = pca_model.transform(df_obs_sum_3_scaled)

print("Percentage of variance explained: ", pca_model.explained_variance_ratio_)
print(np.cumsum(pca_model.explained_variance_ratio_))

# Plot the projected data with colors based on mechanism weights
fig, axis = plt.subplots(1, num_mechs, figsize=(14,7))
for k in range(num_mechs):
    tmp = axis[k].scatter(df_summaries_PCA[:,0],
                          df_summaries_PCA[:,1],
                          c=df_weights_params_sel_train.iloc[:,k], s=35, vmin=0, vmax=1)
    axis[k].scatter(df_obs_sum_1_PCA[:,0],
                    df_obs_sum_1_PCA[:,1],
                    c='red', s=40)
    axis[k].scatter(df_obs_sum_2_PCA[:,0],
                    df_obs_sum_2_PCA[:,1],
                    c='magenta', s=40)
    axis[k].scatter(df_obs_sum_3_PCA[:,0],
                    df_obs_sum_3_PCA[:,1],
                    c='cyan', s=40)
    fig.colorbar(tmp, ax=axis[k])
    if k == 0 :
        axis[k].set_ylabel("PCA 2")
    axis[k].set_xlabel("PCA 1")
fig.suptitle("Data projected onto the two first PCA components")
plt.show()


### PLS analyses

# No need to use df_summaries_ref_table_scaled, because PLSCanonical will scale the data
# (Since everything is scaled, I still use it)

PLSCA_model = PLSCanonical(n_components = 4)
df_summaries_PLSCA, df_weights_params_scaled_PLSCA = PLSCA_model.fit_transform(df_summaries_scaled_sel_train,
                                                                               df_weights_params_scaled_sel_train)
df_obs_sum_1_PLSCA = PLSCA_model.transform(df_obs_sum_1_scaled)
df_obs_sum_2_PLSCA = PLSCA_model.transform(df_obs_sum_2_scaled)
df_obs_sum_3_PLSCA = PLSCA_model.transform(df_obs_sum_3_scaled)


# Plot the projected data with colors based on mechanism weights
fig, axis = plt.subplots(1, num_mechs, figsize=(14,7))
for k in range(num_mechs):
    tmp = axis[k].scatter(df_summaries_PLSCA[:,0],
                          df_summaries_PLSCA[:,1],
                          c=df_weights_params_sel_train.iloc[:,k], s=35, vmin=0, vmax=1)
    axis[k].scatter(df_obs_sum_1_PLSCA[:,0],
                    df_obs_sum_1_PLSCA[:,1],
                    c='red', s=40)
    axis[k].scatter(df_obs_sum_2_PLSCA[:,0],
                    df_obs_sum_2_PLSCA[:,1],
                    c='magenta', s=40)
    axis[k].scatter(df_obs_sum_3_PLSCA[:,0],
                    df_obs_sum_3_PLSCA[:,1],
                    c='cyan', s=40)
    fig.colorbar(tmp, ax=axis[k])
    if k == 0 :
        axis[k].set_ylabel("PLS 2")
    axis[k].set_xlabel("PLS 1")
fig.suptitle("Data projected onto the two first PLS components")
plt.show()


###############################################################################
### Optimize m_try
###############################################################################

min_mtry = 1
max_mtry = df_summaries_scaled_sel_train.shape[1]
step_mtry = 1
mtry_values = np.arange(1,max_mtry+1,step_mtry)
n_jobs = 7

### With the OOB error rate (mean squared error)
oob_error_rates = []
forest_reg_oob = RandomForestRegressor(n_estimators=500,
                                       criterion='mse',
                                       bootstrap=True,
                                       oob_score=True,
                                       n_jobs=n_jobs,
                                       random_state=123)

# Using the negative_mean_squared_error as score function
for i in mtry_values:
    forest_reg_oob.set_params(max_features=i)
    forest_reg_oob.fit(covariates_all_scaled_sel_train, response_all_scaled_sel_train)
    oob_error = mean_squared_error(response_all_sel_train, 
                                    forest_reg_oob.oob_prediction_*std_prior_params,
                                    multioutput='uniform_average', 
                                    squared=True)
    oob_error_rates += [oob_error]
    
plt.plot(mtry_values, oob_error_rates)
plt.xlim(min_mtry, max_mtry)
plt.xlabel("m_try")
plt.ylabel("RF OOB error rate")
plt.legend(loc="upper right")
plt.show()

m_try_oob = mtry_values[list(oob_error_rates).index(np.min(oob_error_rates))]
print("oob_error_rates: ", oob_error_rates)
print("m_try selected with OOB error: ", m_try_oob)

rf_seed = 123
m_try = m_try_oob # Selected by both CV and OOB error

###############################################################################
###!!! Reduce the data space and check performance gain or not with k-NN ABC algo.
###############################################################################

### Below we rely on a k-NN algorithm to determine the ABC posterior,
### we check the accuracy gain or loss when using the reduced number of summaries.

############### Without summary statistic selection ###########################

##### Estimate the parameters #####

# NearestNeighbors does not standardize the data, so I need to use my *_scaled summaries
nearest_neigh = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh.fit(df_summaries_scaled_knn)
distances_obs_1, indices_obs_1 = nearest_neigh.kneighbors(df_obs_sum_1_scaled)
distances_obs_2, indices_obs_2 = nearest_neigh.kneighbors(df_obs_sum_2_scaled)
distances_obs_3, indices_obs_3 = nearest_neigh.kneighbors(df_obs_sum_3_scaled)

# ##### For the 1st observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_1_knn = df_weights_knn.iloc[indices_obs_1[0],:]

print("mean: ", np.array(accepted_weights_obs_1_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_1_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_1_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_1_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_1_knn = df_params_knn.iloc[indices_obs_1[0],:]

counter_tmp = accepted_params_obs_1_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_1_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


##### For the 2nd observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_2_knn = df_weights_knn.iloc[indices_obs_2[0],:]

print("mean: ", np.array(accepted_weights_obs_2_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_2_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_2_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_2_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_2_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_2_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_2_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_2_knn = df_params_knn.iloc[indices_obs_2[0],:]

counter_tmp = accepted_params_obs_2_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_2_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_2_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


##### For the 3rd observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_3_knn = df_weights_knn.iloc[indices_obs_3[0],:]

print("mean: ", np.array(accepted_weights_obs_3_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_3_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_3_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_3_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_3_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_3_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_3_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_3_knn = df_params_knn.iloc[indices_obs_3[0],:]

counter_tmp = accepted_params_obs_3_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_3_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_3_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()
    

###############################################################################
##### Select the summary statistics with random forests RFA with error 
##### minimization in neighborhood of the observed data
###############################################################################

n_estimators = 500
num_neigh_PCA = 100
num_neighbors_KNN = 200
n_repeats_MDA = 10
n_jobs = 7
pred_type = 'individual'

os.chdir(dir_save_results)

# For observation 1

rf_model_unsel = RandomForestRegressor(n_estimators=n_estimators, 
                                       criterion='mse',
                                       max_features=m_try, 
                                       bootstrap=True,
                                       oob_score=False, 
                                       n_jobs=n_jobs,
                                       random_state=rf_seed)

time1 = time.time()
average_RMSE_perResponse_obs1, average_RMSE_total_obs1, num_sel_sums_final_obs1, ranking_RFMDA_local_obs1 = \
    RFMDA_select_summaries(rf_model=rf_model_unsel,
                           covariates_train=df_summaries_scaled_sel_train,
                           responses_scaled_train=df_weights_params_scaled_sel_train,
                           covariates_val=df_summaries_scaled_sel_val, 
                           responses_scaled_val=df_weights_params_scaled_sel_val,
                           covariates_knn=df_summaries_scaled_knn,
                           responses_scaled_knn=df_weights_params_scaled_knn,
                           covariates_obs=df_obs_sum_1_scaled,
                           n_repeats_MDA=n_repeats_MDA, random_state_MDA=123,
                           pvar_min_PCA=0.90, num_neigh_PCA=num_neigh_PCA,
                           num_neighbors_KNN=num_neighbors_KNN,
                           pred_type=pred_type)
time2 = time.time()
print("Time Obs 1 nonRecursif: ", time2 - time1)

pickle.dump(average_RMSE_perResponse_obs1, open("average_RMSE_perResponse_obs1_"+pred_type+example_number+".p", "wb"))
pickle.dump(average_RMSE_total_obs1, open("average_RMSE_total_obs1_"+pred_type+example_number+".p", "wb"))
pickle.dump(ranking_RFMDA_local_obs1, open("ranking_RFMDA_local_obs1_"+pred_type+example_number+".p", "wb"))

# For observation 2

rf_model_unsel = RandomForestRegressor(n_estimators=500, 
                                       criterion='mse',
                                       max_features=m_try,
                                       bootstrap=True,
                                       oob_score=False,
                                       n_jobs=n_jobs,
                                       random_state=rf_seed)

time1 = time.time()
average_RMSE_perResponse_obs2, average_RMSE_total_obs2, num_sel_sums_final_obs2, ranking_RFMDA_local_obs2 = \
    RFMDA_select_summaries(rf_model=rf_model_unsel,
                           covariates_train=df_summaries_scaled_sel_train,
                           responses_scaled_train=df_weights_params_scaled_sel_train,
                           covariates_val=df_summaries_scaled_sel_val,
                           responses_scaled_val=df_weights_params_scaled_sel_val,
                           covariates_knn=df_summaries_scaled_knn,
                           responses_scaled_knn=df_weights_params_scaled_knn,
                           covariates_obs=df_obs_sum_2_scaled,
                           n_repeats_MDA=n_repeats_MDA, random_state_MDA=123,
                           pvar_min_PCA=0.90, num_neigh_PCA=num_neigh_PCA,
                           num_neighbors_KNN=num_neighbors_KNN,
                           pred_type=pred_type)
time2 = time.time()
print("Time Obs 2 nonRecursif: ", time2 - time1)

pickle.dump(average_RMSE_perResponse_obs2, open("average_RMSE_perResponse_obs2_"+pred_type+example_number+".p", "wb"))
pickle.dump(average_RMSE_total_obs2, open("average_RMSE_total_obs2_"+pred_type+example_number+".p", "wb"))
pickle.dump(ranking_RFMDA_local_obs2, open("ranking_RFMDA_local_obs2_"+pred_type+example_number+".p", "wb"))

# For observation 3

rf_model_unsel = RandomForestRegressor(n_estimators=500, 
                                       criterion='mse',
                                       max_features=m_try, 
                                       bootstrap=True,
                                       oob_score=False, 
                                       n_jobs=n_jobs,
                                       random_state=rf_seed)

time1 = time.time()
average_RMSE_perResponse_obs3, average_RMSE_total_obs3, num_sel_sums_final_obs3, ranking_RFMDA_local_obs3 = \
    RFMDA_select_summaries(rf_model=rf_model_unsel,
                           covariates_train=df_summaries_scaled_sel_train,
                           responses_scaled_train=df_weights_params_scaled_sel_train,
                           covariates_val=df_summaries_scaled_sel_val, 
                           responses_scaled_val=df_weights_params_scaled_sel_val,
                           covariates_knn=df_summaries_scaled_knn,
                           responses_scaled_knn=df_weights_params_scaled_knn,
                           covariates_obs=df_obs_sum_3_scaled,
                           n_repeats_MDA=n_repeats_MDA, random_state_MDA=123,
                           pvar_min_PCA=0.90, num_neigh_PCA=num_neigh_PCA,
                           num_neighbors_KNN=num_neighbors_KNN,
                           pred_type=pred_type)
time2 = time.time()
print("Time Obs 3 nonRecursif: ", time2 - time1)

pickle.dump(average_RMSE_perResponse_obs3, open("average_RMSE_perResponse_obs3_"+pred_type+example_number+".p", "wb"))
pickle.dump(average_RMSE_total_obs3, open("average_RMSE_total_obs3_"+pred_type+example_number+".p", "wb"))
pickle.dump(ranking_RFMDA_local_obs3, open("ranking_RFMDA_local_obs3_"+pred_type+example_number+".p", "wb"))

##### KNN results with corresponding summary statistics #####

print("selected summaries obs1: ", ranking_RFMDA_local_obs1)
print("number selected ", len(ranking_RFMDA_local_obs1))

print("selected summaries obs2: ", ranking_RFMDA_local_obs2)
print("number selected ", len(ranking_RFMDA_local_obs2))

print("selected summaries obs3: ", ranking_RFMDA_local_obs3)
print("number selected ", len(ranking_RFMDA_local_obs3))

# For observation 1

df_summaries_scaled_sel_knn_obs1 = df_summaries_scaled_knn.iloc[:,ranking_RFMDA_local_obs1]
df_summaries_scaled_sel_knn_obs2 = df_summaries_scaled_knn.iloc[:,ranking_RFMDA_local_obs2]
df_summaries_scaled_sel_knn_obs3 = df_summaries_scaled_knn.iloc[:,ranking_RFMDA_local_obs3]

df_obs_sum_1_scaled_sel =  df_obs_sum_1_scaled.iloc[:,ranking_RFMDA_local_obs1]
df_obs_sum_2_scaled_sel =  df_obs_sum_2_scaled.iloc[:,ranking_RFMDA_local_obs2]
df_obs_sum_3_scaled_sel =  df_obs_sum_3_scaled.iloc[:,ranking_RFMDA_local_obs3]


##### Estimate the parameters #####

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs1)
distances_obs_1_sel, indices_obs_1_sel = nearest_neigh_sel.kneighbors(df_obs_sum_1_scaled_sel)

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs2)
distances_obs_2_sel, indices_obs_2_sel = nearest_neigh_sel.kneighbors(df_obs_sum_2_scaled_sel)

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs3)
distances_obs_3_sel, indices_obs_3_sel = nearest_neigh_sel.kneighbors(df_obs_sum_3_scaled_sel)

##### For the 1st observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_1_sel_knn = df_weights_knn.iloc[indices_obs_1_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_1_sel_knn = df_params_knn.iloc[indices_obs_1_sel[0],:]

counter_tmp = accepted_params_obs_1_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_sel_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_1_sel_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


##### For the 2nd observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_2_sel_knn = df_weights_knn.iloc[indices_obs_2_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_2_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_2_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_2_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_2_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_2_sel_knn = df_params_knn.iloc[indices_obs_2_sel[0],:]

counter_tmp = accepted_params_obs_2_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_2_sel_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_2_sel_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


##### For the 3rd observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_3_sel_knn = df_weights_knn.iloc[indices_obs_3_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_3_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_3_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_3_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_3_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_3_sel_knn = df_params_knn.iloc[indices_obs_3_sel[0],:]

counter_tmp = accepted_params_obs_3_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_3_sel_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_3_sel_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


###############################################################################
##### Select the summary statistics with random forests RFA with error 
##### minimization in neighborhood of the observed data
###############################################################################

# For observation 1

time1 = time.time()

average_RMSE_perResponse_recursif_obs1, average_RMSE_total_recursif_obs1, eliminated_features_recursif_obs1, recursive_selected_summaries_obs1 =\
    recursiveElimination_RFMDA_select_summaries(covariates_train=df_summaries_scaled_sel_train,
                                                responses_scaled_train=df_weights_params_scaled_sel_train,
                                                covariates_val=df_summaries_scaled_sel_val,
                                                responses_scaled_val=df_weights_params_scaled_sel_val,
                                                covariates_knn=df_summaries_scaled_knn,
                                                responses_scaled_knn=df_weights_params_scaled_knn,
                                                covariates_obs=df_obs_sum_1_scaled,
                                                n_estimators=n_estimators,
                                                max_features='auto',
                                                n_repeats_MDA=n_repeats_MDA,
                                                random_state_MDA=123,
                                                pvar_min_PCA=0.90,
                                                num_neigh_PCA=num_neigh_PCA,
                                                num_neighbors_KNN=num_neighbors_KNN,
                                                pred_type=pred_type,
                                                n_jobs=n_jobs)
time2 = time.time()
print("Time Obs 1 recursif: ", time2 - time1)

pickle.dump(average_RMSE_perResponse_recursif_obs1, open("average_RMSE_perResponse_recursif_obs1_"+pred_type+example_number+".p", "wb"))
pickle.dump(average_RMSE_total_recursif_obs1, open("average_RMSE_total_recursif_obs1_"+pred_type+example_number+".p", "wb"))
pickle.dump(eliminated_features_recursif_obs1, open("eliminated_features_recursif_obs1_"+pred_type+example_number+".p", "wb"))
pickle.dump(recursive_selected_summaries_obs1, open("recursive_selected_summaries_obs1_"+pred_type+example_number+".p", "wb"))

time1 = time.time()

average_RMSE_perResponse_recursif_obs2, average_RMSE_total_recursif_obs2, eliminated_features_recursif_obs2, recursive_selected_summaries_obs2 =\
    recursiveElimination_RFMDA_select_summaries(covariates_train=df_summaries_scaled_sel_train,
                                                responses_scaled_train=df_weights_params_scaled_sel_train,
                                                covariates_val=df_summaries_scaled_sel_val,
                                                responses_scaled_val=df_weights_params_scaled_sel_val,
                                                covariates_knn=df_summaries_scaled_knn,
                                                responses_scaled_knn=df_weights_params_scaled_knn,
                                                covariates_obs=df_obs_sum_2_scaled,
                                                n_estimators=n_estimators,
                                                max_features='auto',
                                                n_repeats_MDA=n_repeats_MDA,
                                                random_state_MDA=123,
                                                pvar_min_PCA=0.90,
                                                num_neigh_PCA=num_neigh_PCA,
                                                num_neighbors_KNN=num_neighbors_KNN,
                                                pred_type=pred_type,
                                                n_jobs=n_jobs)
time2 = time.time()
print("Time Obs 2 recursif: ", time2 - time1)

pickle.dump(average_RMSE_perResponse_recursif_obs2, open("average_RMSE_perResponse_recursif_obs2_"+pred_type+example_number+".p", "wb"))
pickle.dump(average_RMSE_total_recursif_obs2, open("average_RMSE_total_recursif_obs2_"+pred_type+example_number+".p", "wb"))
pickle.dump(eliminated_features_recursif_obs2, open("eliminated_features_recursif_obs2_"+pred_type+example_number+".p", "wb"))
pickle.dump(recursive_selected_summaries_obs2, open("recursive_selected_summaries_obs2_"+pred_type+example_number+".p", "wb"))


time1 = time.time()

average_RMSE_perResponse_recursif_obs3, average_RMSE_total_recursif_obs3, eliminated_features_recursif_obs3, recursive_selected_summaries_obs3 =\
    recursiveElimination_RFMDA_select_summaries(covariates_train=df_summaries_scaled_sel_train,
                                                responses_scaled_train=df_weights_params_scaled_sel_train,
                                                covariates_val=df_summaries_scaled_sel_val,
                                                responses_scaled_val=df_weights_params_scaled_sel_val,
                                                covariates_knn=df_summaries_scaled_knn,
                                                responses_scaled_knn=df_weights_params_scaled_knn,
                                                covariates_obs=df_obs_sum_3_scaled,
                                                n_estimators=n_estimators,
                                                max_features='auto',
                                                n_repeats_MDA=n_repeats_MDA,
                                                random_state_MDA=123,
                                                pvar_min_PCA=0.90,
                                                num_neigh_PCA=num_neigh_PCA,
                                                num_neighbors_KNN=num_neighbors_KNN,
                                                pred_type=pred_type,
                                                n_jobs=n_jobs)

time2 = time.time()
print("Time Obs 3 recursif: ", time2 - time1)

pickle.dump(average_RMSE_perResponse_recursif_obs3, open("average_RMSE_perResponse_recursif_obs3_"+pred_type+example_number+".p", "wb"))
pickle.dump(average_RMSE_total_recursif_obs3, open("average_RMSE_total_recursif_obs3_"+pred_type+example_number+".p", "wb"))
pickle.dump(eliminated_features_recursif_obs3, open("eliminated_features_recursif_obs3_"+pred_type+example_number+".p", "wb"))
pickle.dump(recursive_selected_summaries_obs3, open("recursive_selected_summaries_obs3_"+pred_type+example_number+".p", "wb"))


##### KNN results with corresponding summary statistics #####

print("selected summaries obs1: ", recursive_selected_summaries_obs1)
print("number selected ", len(recursive_selected_summaries_obs1))

print("selected summaries obs2: ", recursive_selected_summaries_obs2)
print("number selected ", len(recursive_selected_summaries_obs2))

print("selected summaries obs3: ", recursive_selected_summaries_obs3)
print("number selected ", len(recursive_selected_summaries_obs3))

# For observation 1

df_summaries_scaled_sel_knn_obs1 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs1]
df_summaries_scaled_sel_knn_obs2 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs2]
df_summaries_scaled_sel_knn_obs3 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs3]

df_obs_sum_1_scaled_sel =  df_obs_sum_1_scaled.iloc[:,recursive_selected_summaries_obs1]
df_obs_sum_2_scaled_sel =  df_obs_sum_2_scaled.iloc[:,recursive_selected_summaries_obs2]
df_obs_sum_3_scaled_sel =  df_obs_sum_3_scaled.iloc[:,recursive_selected_summaries_obs3]


##### Estimate the parameters #####

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs1)
distances_obs_1_sel, indices_obs_1_sel = nearest_neigh_sel.kneighbors(df_obs_sum_1_scaled_sel)

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs2)
distances_obs_2_sel, indices_obs_2_sel = nearest_neigh_sel.kneighbors(df_obs_sum_2_scaled_sel)

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs3)
distances_obs_3_sel, indices_obs_3_sel = nearest_neigh_sel.kneighbors(df_obs_sum_3_scaled_sel)

##### For the 1st observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_1_sel_knn = df_weights_knn.iloc[indices_obs_1_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_1_sel_knn = df_params_knn.iloc[indices_obs_1_sel[0],:]

counter_tmp = accepted_params_obs_1_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_sel_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_1_sel_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


##### For the 2nd observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_2_sel_knn = df_weights_knn.iloc[indices_obs_2_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_2_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_2_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_2_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_2_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_2_sel_knn = df_params_knn.iloc[indices_obs_2_sel[0],:]

counter_tmp = accepted_params_obs_2_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_2_sel_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_2_sel_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()


##### For the 3rd observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_3_sel_knn = df_weights_knn.iloc[indices_obs_3_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_3_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_3_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_3_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_3_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# plot the posteriors
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,0], alpha=0.4, label='pref att weight')
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,1], alpha=0.4, label='rand att weight')
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,2], alpha=0.4, label='triangle formation')
plt.legend()
plt.show()

# recover the parameters
accepted_params_obs_3_sel_knn = df_params_knn.iloc[indices_obs_3_sel[0],:]

counter_tmp = accepted_params_obs_3_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_3_sel_knn.iloc[:,0], alpha=0.4, label='pref att m parameter', bins=range(1,11))
plt.hist(accepted_params_obs_3_sel_knn.iloc[:,1], alpha=0.4, label='rand att m parameter', bins=range(1,11))
plt.legend()
plt.show()