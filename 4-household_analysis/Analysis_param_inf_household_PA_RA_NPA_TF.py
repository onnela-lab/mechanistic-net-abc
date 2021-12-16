# -*- coding: utf-8 -*-
"""
Analysis of the real data: household contact network.
"""

import networkx as nx
import pandas as pd
import numpy as np
import scipy.stats as ss
import time
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import pickle
import multiprocessing

from collections import Counter
from collections import defaultdict
from pkg_resources import resource_filename

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from mechanistic_net_abc.summaries import compute_many_summaries_undirected
from mechanistic_net_abc.mechanisms import preferential_attachment_growth, random_attachment_growth, triangle_formation_node_addition, neg_preferential_attachment_growth
from mechanistic_net_abc.data_generation import data_ref_table_simulation
from mechanistic_net_abc.utility import drop_redundant_features
from mechanistic_net_abc.abc import abc_RSMCABC, distance_euclidean_std
from mechanistic_net_abc.models import mixture_model_simulation
from mechanistic_net_abc.summary_selection import recursiveElimination_RFMDA_select_summaries
from mechanistic_net_abc.summaries import compute_indexed_summaries_undirected
from mechanistic_net_abc.data_reading import read_household_data_filepath

from mechanistic_net_abc.settings import base_dir_household

#######################################################
### Define the general setting
#######################################################

np.random.seed(123)

example_number = "_PA_RA_NPA_TF"
num_cores = max(1,multiprocessing.cpu_count()-1)

### Read the observed data
path_household_data = resource_filename("mechanistic_net_abc", "data/4-household_analysis/household_data/edgelist_g_vstatusok_lcc.txt")
household_network = read_household_data_filepath(path_household_data)

### Info about the observation
num_nodes = household_network.number_of_nodes()
print(num_nodes)
print(household_network.number_of_edges())

nx.draw(household_network, node_size=10)
plt.show()

# Compute the observed summary statistics
obs_summaries = compute_many_summaries_undirected(household_network)
df_obs_summaries = pd.DataFrame([obs_summaries])

# Plot of the degree distribution
degree_sequence = sorted([d for n, d in household_network.degree()], reverse=True)
degreeCount = Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="b")
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d  for d in deg])
ax.set_xticklabels(deg)
plt.show()

###### Directories specification

###!!! Set the base working directory, to change as you like from the settings.py file
dir_base = base_dir_household
os.chdir(dir_base)

### Set the directory for data saving
dir_save_data = dir_base+"/data_"+str(num_nodes)+"nodes"+example_number
if not os.path.exists(dir_save_data):
    os.makedirs(dir_save_data)
    print("Directory created")

# To save the recursive RF rankings
dir_save_rankings = dir_base+"/saved_rankings"
if not os.path.exists(dir_save_rankings):
    os.makedirs(dir_save_rankings)
    print("Directory created")

dir_save_plots = dir_base+"/saved_figures"
if not os.path.exists(dir_save_plots):
    os.makedirs(dir_save_plots)
    print("Directory created")

#######################################################
### Definition of the generative model
#######################################################

### Model definition:
mech1 = preferential_attachment_growth
mech2 = random_attachment_growth
mech3 = neg_preferential_attachment_growth
mech4 = triangle_formation_node_addition

# Seed network definition
num_nodes_seed = 3
num_edges_seed = 2
seed = 123
# Use seed for reproducibility
G_seed = nx.gnm_random_graph(n=num_nodes_seed,
                             m=num_edges_seed,
                             seed=seed)
nx.draw(G_seed)
plt.show()

# Define the mechanisms and associated mechanisms
func_mechanisms = [mech1, mech2, mech3, mech4]
num_mechs = len(func_mechanisms)

max_m_value = num_nodes_seed
prior_m_pref_att = ss.randint(1, max_m_value+1)
prior_m_rand_att = ss.randint(1, max_m_value+1)
prior_m_neg_pref_att = ss.randint(1, max_m_value+1)

prior_args_mechanisms = [{'m':prior_m_pref_att},
                         {'m':prior_m_rand_att},
                         {'m':prior_m_neg_pref_att},
                         {}]
fixed_args_mechanisms = [{'degree_aug':1},
                         {},
                         {'degree_aug':1},
                         {}]
min_weights = [0,0,0,0]
max_weights = [1,1,1,1]

# Compute the prior standard deviation of the parameters (including weights)
std_prior_params = np.sqrt(ss.dirichlet([1]*num_mechs).var().tolist() + [prior_m_pref_att.var()] + [prior_m_rand_att.var()] + [prior_m_neg_pref_att.var()])

# Number of simulated data in the reference table
# (then splitted in training validation and inference)
num_sim = 100000
num_sim_sel_train = 40000
num_sim_sel_val = 10000
num_sim_knn = 50000

# Number of neighbors for K-NN-ABC:
num_neigh = 200

#########################################################################################
### Generate the num_sim reference table
#########################################################################################

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper

# time1 = time.time()
# (df_weights_ref_table,
#  df_params_ref_table,
#  df_summaries_ref_table_init) = data_ref_table_simulation(G_seed = G_seed,
#                                                           num_sim = num_sim,
#                                                           num_nodes = num_nodes,
#                                                           func_mechanisms = func_mechanisms,
#                                                           prior_args_mechanisms = prior_args_mechanisms,
#                                                           fixed_args_mechanisms = fixed_args_mechanisms,
#                                                           num_cores = num_cores,
#                                                           min_weight = min_weights,
#                                                           max_weight = max_weights,
#                                                           many_summaries = True)
# time2 = time.time()
# print("Time to simulate the reference table: {} seconds.".format(time2 - time1))
#
# df_weights_ref_table.to_csv(dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
# df_params_ref_table.to_csv(dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
# df_summaries_ref_table_init.to_csv(dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)

df_weights_ref_table = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/ref_table/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv"))
df_params_ref_table = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/ref_table/df_params_ref_table_size_"+str(num_sim)+example_number+".csv"))
df_summaries_ref_table_init = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/ref_table/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv"))

###############################################################################
#### Drop the redundant features (i.e. no value variability or correlation 1)
###############################################################################

### We discard the summary statistics that present no variability

# For the reference table
df_summaries_ref_table_r = drop_redundant_features(df_summaries_ref_table_init)
# ['num_of_CC', 'num_nodes_LCC', 'num_4cores', 'num_5cores', 'num_6cores',
# 'num_4shells', 'num_5shells', 'num_6shells', 'num_5cliques'] are removed

# For the observed summaries too
nunique = df_summaries_ref_table_init.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique==1].index
df_obs_summaries_r = df_obs_summaries.drop(cols_to_drop, axis=1)

### We keep only one summary statistic per cluster of summaries that present a correction of 1
plt.figure(figsize=(15,10))
correlations = df_summaries_ref_table_r.corr() # small approx error, corrected below
correlations[correlations>1] = 1
correlations[correlations<-1] = -1
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True,
            annot_kws={"size": 7}, vmin=-1, vmax=1)

plt.figure(figsize=(12,5))
dissimilarity = 1 - abs(round(correlations,2))
Z = linkage(squareform(dissimilarity), 'complete')
dendrogram(Z, labels=df_summaries_ref_table_r.columns,
           orientation='top',
           leaf_rotation=90)

threshold = 0 # Search for correlation 1 clusters
labels_clust = fcluster(Z, threshold, criterion='distance')

### Keep one summary statistic per cluster (among pairs that have a correlation of 1)
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(labels_clust):
    cluster_id_to_feature_ids[cluster_id].append(idx)

print(cluster_id_to_feature_ids)

selected_features_clust = [v[0] for v in cluster_id_to_feature_ids.values()]
idx_cols_to_drop_clust = list(set(range(df_summaries_ref_table_r.shape[1])) - set(selected_features_clust))
cols_to_drop_clust = df_summaries_ref_table_r.columns[idx_cols_to_drop_clust]
print(cols_to_drop_clust)
# ['num_edges_LCC', 'avg_deg_connectivity_LCC', 'degree_mean',
#  'edge_connectivity_LCC', 'avg_clustering_coef', 'num_3shells']

# Reduce the reference table
df_summaries_ref_table = df_summaries_ref_table_r.drop(cols_to_drop_clust, axis=1)

# For the observed summaries too
df_obs_sum_1 = df_obs_summaries_r.drop(cols_to_drop_clust, axis=1)

### Save the name of the summaries
name_summaries_unsel = np.array(list(df_summaries_ref_table.columns))

###############################################################################
### Scale the data
###############################################################################

scaler = StandardScaler()
df_summaries_ref_table_scaled = pd.DataFrame(scaler.fit_transform(df_summaries_ref_table),
                                             columns = name_summaries_unsel)
df_obs_sum_1_scaled = pd.DataFrame(scaler.transform(df_obs_sum_1),
                                   columns = name_summaries_unsel)

###############################################################################
### Create the joint responses and scale them
###############################################################################

# Drop response that are fixed if necessary (no prior on the parameter)
df_weights_params_ref_table = pd.concat([df_weights_ref_table,
                                         df_params_ref_table],
                                        axis=1, join="inner")

df_weights_params_ref_table = drop_redundant_features(df_weights_params_ref_table)

df_weights_params_scaled = df_weights_params_ref_table/std_prior_params


###############################################################################
### Split the reference table
###############################################################################

# For summary statistic selection (RF training and permutation importance calculation)
idx_data_sel_train = range(num_sim_sel_train)
df_weights_sel_train = df_weights_ref_table.iloc[idx_data_sel_train]
df_params_sel_train = df_params_ref_table.iloc[idx_data_sel_train]
df_summaries_scaled_sel_train = df_summaries_ref_table_scaled.iloc[idx_data_sel_train]
covariates_all_scaled_sel_train = df_summaries_scaled_sel_train.to_numpy(copy=True)

idx_data_sel_val = range(num_sim_sel_train, num_sim_sel_train+num_sim_sel_val)
df_weights_sel_val = df_weights_ref_table.iloc[idx_data_sel_val]
df_params_sel_val = df_params_ref_table.iloc[idx_data_sel_val]
df_summaries_scaled_sel_val = df_summaries_ref_table_scaled.iloc[idx_data_sel_val]
covariates_all_scaled_sel_val = df_summaries_scaled_sel_val.to_numpy(copy=True)

# For k-NN ABC (only the summaries are needed in the end)
idx_data_knn = range(num_sim_sel_train+num_sim_sel_val, num_sim)
df_weights_knn = df_weights_ref_table.iloc[idx_data_knn]
df_params_knn = df_params_ref_table.iloc[idx_data_knn]
df_summaries_scaled_knn = df_summaries_ref_table_scaled.iloc[idx_data_knn]
covariates_all_scaled_knn = df_summaries_scaled_knn.to_numpy(copy=True)

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
### Data visualization through PCA components
###############################################################################

pca_model = PCA(n_components = 6)
df_summaries_PCA = pca_model.fit_transform(df_summaries_scaled_sel_train)
df_obs_sum_1_PCA = pca_model.transform(df_obs_sum_1_scaled)

print("Percentage of variance explained: ", pca_model.explained_variance_ratio_)
explained_variance_PCA = [ round(var_p*100,1) for var_p in pca_model.explained_variance_ratio_ ]

np.sum(explained_variance_PCA)
np.sum( pca_model.explained_variance_ratio_)

# Plot the projected data to check that this model is well specified
x_axis=0
y_axis=1
fig, axis = plt.subplots(1, 1, figsize=(5,6))
axis.scatter(df_summaries_PCA[:,x_axis],
             df_summaries_PCA[:,y_axis],
             c='tab:blue', s=35)
axis.scatter(df_obs_sum_1_PCA[:,x_axis],
             df_obs_sum_1_PCA[:,y_axis],
             c='tab:red', s=40)
axis.set_xlabel("PCA "+str(x_axis+1)+" ("+str(explained_variance_PCA[x_axis])+"%)", size=13)
axis.set_ylabel("PCA "+str(y_axis+1)+" ("+str(explained_variance_PCA[y_axis])+"%)", size=13)
plt.savefig(dir_save_plots+"/PCA_"+str(x_axis)+"_"+str(y_axis)+example_number+"_household.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/PCA_"+str(x_axis)+"_"+str(y_axis)+example_number+"_household.eps", bbox_inches='tight')
plt.show()

x_axis=2
y_axis=3
fig, axis = plt.subplots(1, 1, figsize=(5,6))
axis.scatter(df_summaries_PCA[:,x_axis],
             df_summaries_PCA[:,y_axis],
             c='tab:blue', s=35)
axis.scatter(df_obs_sum_1_PCA[:,x_axis],
             df_obs_sum_1_PCA[:,y_axis],
             c='tab:red', s=40)
axis.set_xlabel("PCA "+str(x_axis+1)+" ("+str(explained_variance_PCA[x_axis])+"%)", size=13)
axis.set_ylabel("PCA "+str(y_axis+1)+" ("+str(explained_variance_PCA[y_axis])+"%)", size=13)
plt.savefig(dir_save_plots+"/PCA_"+str(x_axis)+"_"+str(y_axis)+example_number+"_household.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/PCA_"+str(x_axis)+"_"+str(y_axis)+example_number+"_household.eps", bbox_inches='tight')
plt.show()

x_axis=4
y_axis=5
fig, axis = plt.subplots(1, 1, figsize=(5,6))
axis.scatter(df_summaries_PCA[:,x_axis],
             df_summaries_PCA[:,y_axis],
             c='tab:blue', s=35)
axis.scatter(df_obs_sum_1_PCA[:,x_axis],
             df_obs_sum_1_PCA[:,y_axis],
             c='tab:red', s=40)
axis.set_xlabel("PCA "+str(x_axis+1)+" ("+str(explained_variance_PCA[x_axis])+"%)", size=13)
axis.set_ylabel("PCA "+str(y_axis+1)+" ("+str(explained_variance_PCA[y_axis])+"%)", size=13)
plt.savefig(dir_save_plots+"/PCA_"+str(x_axis)+"_"+str(y_axis)+example_number+"_household.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/PCA_"+str(x_axis)+"_"+str(y_axis)+example_number+"_household.eps", bbox_inches='tight')
plt.show()


# Plot the projected data with colors based on mechanism weights
fig, axis = plt.subplots(1, num_mechs, figsize=(20,6))
for k in range(num_mechs):
    tmp = axis[k].scatter(df_summaries_PCA[:,0],
                          df_summaries_PCA[:,1],
                          c=df_weights_params_sel_train.iloc[:,k], s=35, vmin=0, vmax=1)
    axis[k].scatter(df_obs_sum_1_PCA[:,0],
                    df_obs_sum_1_PCA[:,1],
                    c='tab:red', s=40)
    fig.colorbar(tmp, ax=axis[k])
    if k == 0 :
        axis[k].set_ylabel("PCA 2", size=12)
    axis[k].set_xlabel("PCA 1", size=12)
plt.show()

fig, axis = plt.subplots(1, num_mechs, figsize=(20,6))
for k in range(num_mechs):
    tmp = axis[k].scatter(df_summaries_PCA[:,2],
                          df_summaries_PCA[:,3],
                          c=df_weights_params_sel_train.iloc[:,k], s=35, vmin=0, vmax=1)
    axis[k].scatter(df_obs_sum_1_PCA[:,2],
                    df_obs_sum_1_PCA[:,3],
                    c='tab:red', s=40)
    fig.colorbar(tmp, ax=axis[k])
    if k == 0 :
        axis[k].set_ylabel("PCA 4")
    axis[k].set_xlabel("PCA 3")
plt.show()

fig, axis = plt.subplots(1, num_mechs, figsize=(20,6))
for k in range(num_mechs):
    tmp = axis[k].scatter(df_summaries_PCA[:,4],
                          df_summaries_PCA[:,5],
                          c=df_weights_params_sel_train.iloc[:,k], s=35, vmin=0, vmax=1)
    axis[k].scatter(df_obs_sum_1_PCA[:,4],
                    df_obs_sum_1_PCA[:,5],
                    c='tab:red', s=40)
    fig.colorbar(tmp, ax=axis[k])
    if k == 0 :
        axis[k].set_ylabel("PCA 6")
    axis[k].set_xlabel("PCA 5")
plt.show()


###############################################################################
### For the naive selection method: RF MDA + importance visualization
### optimize the m_try RF parameter based on out-of-bag score or 5-fold cross-val.
###############################################################################

###!!! Uncomment to rerun, else use the results below

### With cross validation and mean squared error
# min_mtry = 1
# max_mtry = df_summaries_scaled_sel_train.shape[1]
# step_mtry = 1
# mtry_values = np.arange(1,max_mtry+1,step_mtry)
# param_grid = [{'max_features':mtry_values}]
# forest_reg_cv = RandomForestRegressor(n_estimators=500,
#                                       criterion='mse',
#                                       bootstrap=True,
#                                       oob_score=False,
#                                       n_jobs=num_cores,
#                                       random_state=123)
# grid_search = GridSearchCV(forest_reg_cv, param_grid, cv=5,
#                             scoring='neg_mean_squared_error',
#                             return_train_score=False)
# grid_search.fit(covariates_all_scaled_sel_train, response_all_scaled_sel_train)

# grid_search.cv_results_.keys()

# print(grid_search.cv_results_["mean_test_score"])
# print(grid_search.cv_results_["rank_test_score"])
# m_try_CV = grid_search.best_params_["max_features"]
# print("m_try selected with CV: ", m_try_CV)


# ### With the out-of-bag error rate (mean squared error)
# oob_error_rates = []
# forest_reg_oob = RandomForestRegressor(n_estimators=500,
#                                         criterion='mse',
#                                         bootstrap=True,
#                                         oob_score=True,
#                                         n_jobs=num_cores,
#                                         random_state=123)

# # Using the negative_mean_squared_error as score function
# for i in mtry_values:
#     forest_reg_oob.set_params(max_features=i)
#     forest_reg_oob.fit(covariates_all_scaled_sel_train, response_all_scaled_sel_train)
#     oob_error = mean_squared_error(response_all_sel_train,
#                                     forest_reg_oob.oob_prediction_*std_prior_params,
#                                     multioutput='uniform_average',
#                                     squared=True)
#     oob_error_rates += [oob_error]

# plt.plot(mtry_values, oob_error_rates)
# plt.xlim(min_mtry, max_mtry)
# plt.xlabel("m_try")
# plt.ylabel("RF OOB error rate")
# plt.legend(loc="upper right")
# plt.show()

# m_try_oob = mtry_values[list(oob_error_rates).index(np.min(oob_error_rates))]
# print("oob_error_rates: ", oob_error_rates)
# print("m_try selected with OOB error: ", m_try_oob)


###############################################################################
### Naive selection of summary stistics: train a RF (multi-output with scaled response)
###############################################################################

# Set the seed for reproducibility
rf_seed = 123

# m_try = m_try_CV  # 19
# m_try = m_try_oob # 21
m_try = 19 # Selected by 5-fold cross-validation

### Use random forest importance to select a subset of summary statistics

rf_model_unsel = RandomForestRegressor(n_estimators=500,
                                       criterion='mse',
                                       max_features=m_try,
                                       bootstrap=True,
                                       oob_score=True,
                                       n_jobs=num_cores,
                                       random_state=rf_seed)

# Train the random forest
rf_model_unsel.fit(covariates_all_scaled_sel_train, response_all_scaled_sel_train)

print("OOB RF score: ", rf_model_unsel.oob_score_)

# Compute a prediction error rate on oob predictions
print(mean_squared_error(response_all_sel_train, rf_model_unsel.oob_prediction_*std_prior_params,
                         multioutput='raw_values', squared=True))
print(mean_squared_error(response_all_sel_train, rf_model_unsel.oob_prediction_*std_prior_params,
                         multioutput='uniform_average', squared=True))

plt.plot(response_all_sel_train[:,0],rf_model_unsel.oob_prediction_[:,0]*std_prior_params[0], 'o')
plt.show()
plt.plot(response_all_sel_train[:,1],rf_model_unsel.oob_prediction_[:,1]*std_prior_params[1], 'o')
plt.show()
plt.plot(response_all_sel_train[:,2],rf_model_unsel.oob_prediction_[:,2]*std_prior_params[2], 'o')
plt.show()
plt.plot(response_all_sel_train[:,3],rf_model_unsel.oob_prediction_[:,3]*std_prior_params[3], 'o')
plt.show()

mPA_truth_vs_pred_scaled = pd.DataFrame(data={'m_PA':response_all_sel_train[:,4], 'm_PA_hat':rf_model_unsel.oob_prediction_[:,4]*std_prior_params[4]})
mRA_truth_vs_pred_scaled = pd.DataFrame(data={'m_RA':response_all_sel_train[:,5], 'm_RA_hat':rf_model_unsel.oob_prediction_[:,5]*std_prior_params[5]})
mNPA_truth_vs_pred_scaled = pd.DataFrame(data={'m_NPA':response_all_sel_train[:,6], 'm_NPA_hat':rf_model_unsel.oob_prediction_[:,6]*std_prior_params[6]})

sns.boxplot(x="m_PA", y="m_PA_hat", data=mPA_truth_vs_pred_scaled)
plt.show()
sns.boxplot(x="m_RA", y="m_RA_hat", data=mRA_truth_vs_pred_scaled)
plt.show()
sns.boxplot(x="m_NPA", y="m_NPA_hat", data=mNPA_truth_vs_pred_scaled)
plt.show()


# Compute a prediction error rate on oob predictions with validation set
pred_rf_model_val = rf_model_unsel.predict(covariates_all_scaled_sel_val)
print(mean_squared_error(response_all_sel_val, pred_rf_model_val*std_prior_params,
                         multioutput='raw_values', squared=True))
print(mean_squared_error(response_all_sel_val,  pred_rf_model_val*std_prior_params,
                         multioutput='uniform_average', squared=True))

plt.plot(response_all_sel_val[:,0],pred_rf_model_val[:,0]*std_prior_params[0], 'o')
plt.show()
plt.plot(response_all_sel_val[:,1],pred_rf_model_val[:,1]*std_prior_params[1], 'o')
plt.show()
plt.plot(response_all_sel_val[:,2],pred_rf_model_val[:,2]*std_prior_params[2], 'o')
plt.show()
plt.plot(response_all_sel_val[:,3],pred_rf_model_val[:,3]*std_prior_params[3], 'o')
plt.show()

mPA_truth_vs_pred_scaled_val = pd.DataFrame(data={'m_PA':response_all_sel_val[:,4], 'm_PA_hat':pred_rf_model_val[:,4]*std_prior_params[4]})
mRA_truth_vs_pred_scaled_val = pd.DataFrame(data={'m_RA':response_all_sel_val[:,5], 'm_RA_hat':pred_rf_model_val[:,5]*std_prior_params[5]})
mNPA_truth_vs_pred_scaled_val = pd.DataFrame(data={'m_NPA':response_all_sel_val[:,6], 'm_NPA_hat':pred_rf_model_val[:,6]*std_prior_params[6]})

sns.boxplot(x="m_PA", y="m_PA_hat", data=mPA_truth_vs_pred_scaled_val)
plt.show()
sns.boxplot(x="m_RA", y="m_RA_hat", data=mRA_truth_vs_pred_scaled_val)
plt.show()
sns.boxplot(x="m_NPA", y="m_NPA_hat", data=mNPA_truth_vs_pred_scaled_val)
plt.show()

# Predict the predicted response on the observed data
print(rf_model_unsel.predict(df_obs_sum_1_scaled)*std_prior_params)

###############################################################################
### Recover the Mean Decrease of Accuracy: with RF trained on scaled responses
###############################################################################

### Deduce the permutation importance for feature evaluation (Mean Decrease of Accuracy)

res_permutation_importance_unsel = \
    permutation_importance(rf_model_unsel, covariates_all_scaled_sel_val, response_all_scaled_sel_val,
                           scoring='neg_mean_squared_error',
                           n_repeats=10, random_state=123)

importance_MDA = res_permutation_importance_unsel.importances_mean
sorted_index_MDA = res_permutation_importance_unsel.importances_mean.argsort()

for idx in sorted_index_MDA:
    print((name_summaries_unsel[idx],
           res_permutation_importance_unsel.importances_mean[idx]))

### Graphical representation for the MDA

# Plot only the num_features_to_plot best summaries
num_features_to_plot = 31
# num_features_to_plot = df_summaries_ref_table.shape[1]
y_ticks = np.arange(0, num_features_to_plot)
fig, ax = plt.subplots()
ax.barh(y_ticks, importance_MDA[sorted_index_MDA[np.arange(-num_features_to_plot,0)]])
ax.set_yticks(y_ticks)
ax.set_yticklabels(name_summaries_unsel[sorted_index_MDA[np.arange(-num_features_to_plot,0)]])
ax.set_title("Random Forest Feature Importances (MDA)")
fig.tight_layout()
plt.show()


###############################################################################
### Decide on the number of summary statistics to keep
###############################################################################

### Based on MDA, we keep the first five best summary statistics
# sorted_index_MDA = np.array([21, 30,  6,  8,  9, 17, 16, 29, 10,  3, 28, 15, 23, 22,  2,  5, 12, 11,  0, 27, 19, 18, 13, 26, 25,  1, 20, 24,  7, 14,  4])
num_summaries_sel = 16
idx_summaries_rfMDA_sel = sorted_index_MDA[np.arange(-num_summaries_sel,0)]
print(name_summaries_unsel[idx_summaries_rfMDA_sel])


###############################################################################
### Perform summary statistic selection with recursive selection technique
###############################################################################

n_estimators = 500
num_neigh_PCA = 100
num_neighbors_KNN = 200
n_repeats_MDA = 10
n_jobs = num_cores
pred_type = 'individual'

# For the only observation

obs_idx = 1

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# time1 = time.time()

# average_RMSE_perResponse_recursif_obs1, average_RMSE_total_recursif_obs1, eliminated_features_recursif_obs1, recursive_selected_summaries_obs1 =\
#     recursiveElimination_RFMDA_select_summaries(covariates_train=df_summaries_scaled_sel_train,
#                                                 responses_scaled_train=df_weights_params_scaled_sel_train,
#                                                 covariates_val=df_summaries_scaled_sel_val,
#                                                 responses_scaled_val=df_weights_params_scaled_sel_val,
#                                                 covariates_knn=df_summaries_scaled_knn,
#                                                 responses_scaled_knn=df_weights_params_scaled_knn,
#                                                 covariates_obs=df_obs_sum_1_scaled,
#                                                 n_estimators=n_estimators,
#                                                 max_features='auto',
#                                                 n_repeats_MDA=n_repeats_MDA,
#                                                 random_state_MDA=123,
#                                                 pvar_min_PCA=0.90,
#                                                 num_neigh_PCA=num_neigh_PCA,
#                                                 num_neighbors_KNN=num_neighbors_KNN,
#                                                 pred_type=pred_type,
#                                                 n_jobs=n_jobs)
# time2 = time.time()
# print("Time recursive selection: ", time2 - time1)

# pickle.dump(average_RMSE_perResponse_recursif_obs1, open(dir_save_rankings+"/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(average_RMSE_total_recursif_obs1, open(dir_save_rankings+"/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(eliminated_features_recursif_obs1, open(dir_save_rankings+"/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(recursive_selected_summaries_obs1, open(dir_save_rankings+"/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))

average_RMSE_perResponse_recursif_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/4-household_analysis/rankings/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
average_RMSE_total_recursif_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/4-household_analysis/rankings/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
eliminated_features_recursif_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/4-household_analysis/rankings/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
recursive_selected_summaries_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/4-household_analysis/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))

print("Number of selected summaries: ", len(recursive_selected_summaries_obs1))
print("Selected summaries: ", name_summaries_unsel[recursive_selected_summaries_obs1])

# Visualization of the local errors (plus parameter specific)
average_RMSE_weights_recursif_obs1 = np.array(average_RMSE_perResponse_recursif_obs1)[:,0:num_mechs].mean(axis=1)
plt.plot(np.arange(0,len(name_summaries_unsel)), average_RMSE_total_recursif_obs1, label="ARMSE combine params")
for i in range(7):
    plt.plot(np.arange(0,len(name_summaries_unsel)), np.array(average_RMSE_perResponse_recursif_obs1)[:,i], label=df_weights_params_scaled_knn.columns[i])
plt.plot(np.arange(0,len(name_summaries_unsel)), average_RMSE_weights_recursif_obs1, label="ARMSE all weights")
plt.legend()
plt.show()


###############################################################################
### Perform inference with K-NN-ABC algorithm
###############################################################################

###############################################################################
###############!!! Without summary statistic selection ########################

sel_type = "unselected_summaries"
pred_alg = "knn"

##### Estimate the parameters #####

nearest_neigh = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh.fit(df_summaries_scaled_knn)
distances_obs_1, indices_obs_1 = nearest_neigh.kneighbors(df_obs_sum_1_scaled)

##### For the observed network ######

# Recover the weights
accepted_weights_obs_1_knn = df_weights_knn.iloc[indices_obs_1[0],:]

print("mean: ", np.array(accepted_weights_obs_1_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_knn.apply(np.quantile, q=(0.25, 0.75)))

correla = accepted_weights_obs_1_knn.corr()
print(correla)

# Plot the posteriors
plt.hist(accepted_weights_obs_1_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_1_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_1_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{NPA}$')
plt.hist(accepted_weights_obs_1_knn.iloc[:,3], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{NPA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_4", kde=True,
             color='tab:red', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper right', prop={'size': 13})
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint plots
sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_2", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_3", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[2,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{NPA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_1_knn = df_params_knn.iloc[indices_obs_1[0],:]

counter_tmp = accepted_params_obs_1_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,4.5))
plt.hist(accepted_params_obs_1_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,4.5))
plt.hist(accepted_params_obs_1_knn.iloc[:,2], alpha=0.5, label='$m_{NPA}$', bins=np.arange(0.5,4.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

###############################################################################
###############!!! With naive summary statistic selection #####################

sel_type = "naive_RFMDA"
pred_alg = "knn"

idx_selection_from_rf = idx_summaries_rfMDA_sel
df_summaries_scaled_sel_knn = df_summaries_scaled_knn.iloc[:,idx_selection_from_rf]

df_obs_sum_1_scaled_sel_naive =  df_obs_sum_1_scaled.iloc[:,idx_selection_from_rf]

##### Estimate the parameters #####

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn)
distances_obs_1_sel, indices_obs_1_sel = nearest_neigh_sel.kneighbors(df_obs_sum_1_scaled_sel_naive)

##### For the observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_1_naive_knn = df_weights_knn.iloc[indices_obs_1_sel[0],:]

correla = accepted_weights_obs_1_naive_knn.corr()
print(correla)

print("mean: ", np.array(accepted_weights_obs_1_naive_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_naive_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_naive_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_naive_knn.apply(np.quantile, q=(0.25, 0.75)))

# Plot the posteriors
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{NPA}$')
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,3], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{NPA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_4", kde=True,
             color='tab:red', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper right', prop={'size': 13})
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint plots
sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_2", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_3", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[2,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{NPA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_1_naive_knn = df_params_knn.iloc[indices_obs_1_sel[0],:]

counter_tmp = accepted_params_obs_1_naive_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_naive_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,4.5))
plt.hist(accepted_params_obs_1_naive_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,4.5))
plt.hist(accepted_params_obs_1_naive_knn.iloc[:,2], alpha=0.5, label='$m_{NPA}$', bins=np.arange(0.5,4.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

###############################################################################
###############!!! With recursive summary statistic selection #################

sel_type = "recursive_RFMDA"
pred_alg = "knn"

# Reduce the reference table

df_summaries_scaled_sel_knn_obs1 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs1]

df_obs_sum_1_scaled_sel =  df_obs_sum_1_scaled.iloc[:,recursive_selected_summaries_obs1]


##### Estimate the parameters #####

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn_obs1)
distances_obs_1_sel, indices_obs_1_sel = nearest_neigh_sel.kneighbors(df_obs_sum_1_scaled_sel)

##### For the observed network ######

# recover the accepted parameters, here the weights
accepted_weights_obs_1_sel_knn = df_weights_knn.iloc[indices_obs_1_sel[0],:]

correla = accepted_weights_obs_1_sel_knn.corr()
print(correla)

print("mean: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

# Plot the posteriors
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{NPA}$')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,3], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{NPA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_4", kde=True,
             color='tab:red', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper right', prop={'size': 13})
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint plots
sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_2", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_3", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[2,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{NPA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()


# Recover the parameters
accepted_params_obs_1_sel_knn = df_params_knn.iloc[indices_obs_1_sel[0],:]

counter_tmp = accepted_params_obs_1_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_sel_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,4.5))
plt.hist(accepted_params_obs_1_sel_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,4.5))
plt.hist(accepted_params_obs_1_sel_knn.iloc[:,2], alpha=0.5, label='$m_{NPA}$', bins=np.arange(0.5,4.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()


############################## SMC-ABC ########################################

###############################################################################
###!!! Run SMC-ABC without summary statistic selection
###############################################################################

sel_type = "unselected_summaries"
pred_alg = "SMC"

###!!! It is too expensive to run SMC without selection (time limit 7 days)

# Compute standardization values on the reference table we already have
std_values_unsel = pd.DataFrame( [df_summaries_ref_table.apply(np.std)] )
distance_args_unsel = {'std_values':std_values_unsel}

min_weight = min_weights
max_weight = max_weights
alpha = 0.1
scale_factor = 2
weight_perturbation="Gaussian"
num_acc_sim = 200
distance_func = distance_euclidean_std
distance_args = distance_args_unsel
sel_sum_names = None # We do not select summaries here

threshold_init = 50 # A large threshold at least
threshold_final = 0 # We stop the algorithm after max_sim simulated data
max_sim = 50000

# Set the directory to same the SMC-ABC results
# sel_type = "unselected_summaries"
# dir_save_results = "E:/postdoc/mechanistic_net_abc/mechanistic_net_abc/4-household_analysis/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
# if not os.path.exists(dir_save_results):
#     os.makedirs(dir_save_results)
#     print("Directory created")

###### For the observed network ######
obs_idx = 1
df_obs_sum_for_RABC = df_obs_sum_1

# time1 = time.time()
# (df_weights_RABC_obs,
#   df_params_RABC_obs,
#   df_dist_acc_RABC_obs,
#   sim_count_total_obs,
#   threshold_values_obs) = abc_RSMCABC(G_seed=G_seed,
#                                       num_nodes=num_nodes,
#                                       func_mechanisms=func_mechanisms,
#                                       prior_args_mechanisms=prior_args_mechanisms,
#                                       fixed_args_mechanisms=fixed_args_mechanisms,
#                                       min_weight=min_weights,
#                                       max_weight=max_weights,
#                                       threshold_init=threshold_init,
#                                       threshold_final=threshold_final,
#                                       alpha=alpha,
#                                       scale_factor=scale_factor,
#                                       weight_perturbation=weight_perturbation,
#                                       num_acc_sim=num_acc_sim,
#                                       df_observed_summaries=df_obs_sum_for_RABC,
#                                       distance_func=distance_func,
#                                       distance_args=distance_args,
#                                       sel_sum_names=sel_sum_names,
#                                       max_sim=max_sim)

# time2 = time.time()
# print("Time ABC-RSMC: ", time2 - time1)
# print("Total number of simulations: ", sim_count_total_obs)

# df_weights_RABC_obs.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)
# df_params_RABC_obs.to_csv(dir_save_results+"/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)
# df_dist_acc_RABC_obs.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
correla = df_weights_RABC_obs.corr()
print(correla)

print("mean: ", np.array(df_weights_RABC_obs.apply(np.mean)))
print("median: ", np.array(df_weights_RABC_obs.apply(np.median)))

print("95% credible intervals: ", df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

# Plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{NPA}$')
plt.hist(df_weights_RABC_obs.iloc[:,3], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{NPA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_4", kde=True,
              color='tab:red', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper right', prop={'size': 13})
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint plots
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_3", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[2,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{NPA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,4.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,4.5))
plt.hist(df_params_RABC_obs.iloc[:,2], alpha=0.5, label='$m_{NPA}$', bins=np.arange(0.5,4.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()



###############################################################################
###!!! Run the SMC-ABC algorithm on the summary statistics selected naively
###############################################################################

sel_type = "naive_RFMDA"
pred_alg = "SMC"
idx_selection_from_rf = idx_summaries_rfMDA_sel

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# Set the directory to same the SMC-ABC results
# dir_save_results = "E:/postdoc/mechanistic_net_abc/mechanistic_net_abc/4-household_analysis/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
# if not os.path.exists(dir_save_results):
#     os.makedirs(dir_save_results)
#     print("Directory created")

# SMC-ABC parameters
std_values_sel = pd.DataFrame( [df_summaries_ref_table.iloc[:,idx_selection_from_rf].apply(np.std)] )
distance_args_sel = {'std_values':std_values_sel}

min_weight = min_weights
max_weight = max_weights
alpha = 0.1
scale_factor = 2
weight_perturbation="Gaussian"
num_acc_sim = 200
distance_func = distance_euclidean_std
distance_args = distance_args_sel
sel_sum_names = list(name_summaries_unsel[idx_selection_from_rf])

threshold_init = 50 # A large threshold at least
threshold_final = 0 # We stop the algorithm after max_sim simulated data
max_sim = 50000


###### For the observed network ######
obs_idx = 1
df_obs_sum_for_RABC = df_obs_sum_1[sel_sum_names]


# time1 = time.time()

# (df_weights_RABC_obs,
#   df_params_RABC_obs,
#   df_dist_acc_RABC_obs,
#   sim_count_total_obs,
#   threshold_values_obs) = abc_RSMCABC(G_seed=G_seed,
#                                       num_nodes=num_nodes,
#                                       func_mechanisms=func_mechanisms,
#                                       prior_args_mechanisms=prior_args_mechanisms,
#                                       fixed_args_mechanisms=fixed_args_mechanisms,
#                                       min_weight=min_weights,
#                                       max_weight=max_weights,
#                                       threshold_init=threshold_init,
#                                       threshold_final=threshold_final,
#                                       alpha=alpha,
#                                       scale_factor=scale_factor,
#                                       weight_perturbation=weight_perturbation,
#                                       num_acc_sim=num_acc_sim,
#                                       df_observed_summaries=df_obs_sum_for_RABC,
#                                       distance_func=distance_func,
#                                       distance_args=distance_args,
#                                       sel_sum_names=sel_sum_names,
#                                       max_sim=max_sim)

# time2 = time.time()
# print("Time ABC-RSMC: ", time2 - time1)
# print("Total number of simulations: ", sim_count_total_obs)

# df_weights_RABC_obs.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)
# df_params_RABC_obs.to_csv(dir_save_results+"/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)
# df_dist_acc_RABC_obs.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_naive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_naive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_naive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
correla = df_weights_RABC_obs.corr()
print(correla)

print("mean: ", np.array(df_weights_RABC_obs.apply(np.mean)))
print("median: ", np.array(df_weights_RABC_obs.apply(np.median)))

print("95% credible intervals: ", df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

# Plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{NPA}$')
plt.hist(df_weights_RABC_obs.iloc[:,3], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{NPA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_4", kde=True,
             color='tab:red', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper right', prop={'size': 13})
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint plots
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_3", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[2,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{NPA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,4.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,4.5))
plt.hist(df_params_RABC_obs.iloc[:,2], alpha=0.5, label='$m_{NPA}$', bins=np.arange(0.5,4.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()


###############################################################################
###!!! Run the SMC-ABC algorithm on the summary statistics selected with recursive method
###############################################################################

sel_type = "recursive_RFMDA"
pred_alg = "SMC"
pred_type = 'individual'

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# Set the directory to same the SMC-ABC results
# dir_save_results = "E:/postdoc/mechanistic_net_abc/mechanistic_net_abc/4-household_analysis/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
# if not os.path.exists(dir_save_results):
#     os.makedirs(dir_save_results)
#     print("Directory created")

###### For the observed network ######

# Load the selected summaries with recursive method
obs_idx = 1
recursive_selected_summaries_obs = pickle.load(open(resource_filename("mechanistic_net_abc", "data/4-household_analysis/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))

# SMC-ABC parameters

min_weight = min_weights
max_weight = max_weights
alpha = 0.1
scale_factor = 2
weight_perturbation="Gaussian"
num_acc_sim = 200
distance_func = distance_euclidean_std
std_values_sel = pd.DataFrame( [df_summaries_ref_table.iloc[:,recursive_selected_summaries_obs].apply(np.std)] )
distance_args_sel = {'std_values':std_values_sel}
distance_args = distance_args_sel
sel_sum_names = list(name_summaries_unsel[recursive_selected_summaries_obs])

threshold_init = 50 # A large threshold at least
threshold_final = 0 # We stop the algorithm after max_sim simulated data
max_sim = 50000

df_obs_sum_for_RABC = df_obs_sum_1[sel_sum_names]

# time1 = time.time()
# (df_weights_RABC_obs,
#   df_params_RABC_obs,
#   df_dist_acc_RABC_obs,
#   sim_count_total_obs,
#   threshold_values_obs) = abc_RSMCABC(G_seed=G_seed,
#                                       num_nodes=num_nodes,
#                                       func_mechanisms=func_mechanisms,
#                                       prior_args_mechanisms=prior_args_mechanisms,
#                                       fixed_args_mechanisms=fixed_args_mechanisms,
#                                       min_weight=min_weights,
#                                       max_weight=max_weights,
#                                       threshold_init=threshold_init,
#                                       threshold_final=threshold_final,
#                                       alpha=alpha,
#                                       scale_factor=scale_factor,
#                                       weight_perturbation=weight_perturbation,
#                                       num_acc_sim=num_acc_sim,
#                                       df_observed_summaries=df_obs_sum_for_RABC,
#                                       distance_func=distance_func,
#                                       distance_args=distance_args,
#                                       sel_sum_names=sel_sum_names,
#                                       max_sim=max_sim)

# time2 = time.time()
# print("Time ABC-RSMC: ", time2 - time1)
# print("Total number of simulations: ", sim_count_total_obs)

# df_weights_RABC_obs.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)
# df_params_RABC_obs.to_csv(dir_save_results+"/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)
# df_dist_acc_RABC_obs.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv", index=False)

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_recursive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_recursive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_recursive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print("mean: ", np.array(df_weights_RABC_obs.apply(np.mean)))
print("median: ", np.array(df_weights_RABC_obs.apply(np.median)))

print("95% credible intervals: ", df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

correla = df_weights_RABC_obs.corr()
print(correla)

# Plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{NPA}$')
plt.hist(df_weights_RABC_obs.iloc[:,3], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{NPA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_4", kde=True,
             color='tab:red', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper right', prop={'size': 13})
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint plots
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{NPA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_NPA_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_3", y="weight_mech_4",
            color='tab:blue')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[2,3],3)), fontsize=13)
plt.xlabel(r'$\alpha_{NPA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wNPA_TF_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,4.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,4.5))
plt.hist(df_params_RABC_obs.iloc[:,2], alpha=0.5, label='$m_{NPA}$', bins=np.arange(0.5,4.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+example_number+".eps", bbox_inches='tight')
plt.show()



###############################################################################
### Posterior checking for SMC with recursive selection
###############################################################################

### Can we reproduce the observed data from the resulting parameter estimates?

# Let's generate some simulated data from the predictive posterior distribution

# Obs data
household_network_sums_sel = compute_indexed_summaries_undirected(household_network, sel_sum_names)
df_household_network_sums_sel = pd.DataFrame([household_network_sums_sel])

np.random.seed(seed=123)

num_samples_post_pred = 10
list_G_sim_PP = []
for i in range(num_samples_post_pred):
    sampled_idx = random.randint(0, num_neigh-1)
    sampled_weights = df_weights_RABC_obs.iloc[sampled_idx,:].to_list()
    sampled_params = df_params_RABC_obs.iloc[sampled_idx,:].to_list()
    G_sim = mixture_model_simulation(G_seed=G_seed, num_nodes=num_nodes,
                                      weights=sampled_weights,
                                      func_mechanisms=func_mechanisms,
                                      args_mechanisms=[{'m':int(sampled_params[0]),'degree_aug':1},
                                                      {'m':int(sampled_params[1])},
                                                      {'m':int(sampled_params[2]),'degree_aug':1},
                                                      {}]
                                      )
    G_sim_sums_sel = compute_indexed_summaries_undirected(G_sim, sel_sum_names)
    list_G_sim_PP += [G_sim_sums_sel]
df_G_sim_sums_sel_PP = pd.DataFrame(list_G_sim_PP)

household_network.number_of_edges()
nx.draw(household_network, node_size=10)
plt.savefig(dir_save_plots+"/real_household_data_representation.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/real_household_data_representation.eps", bbox_inches='tight')
plt.show()

G_sim.number_of_edges()
nx.draw(G_sim, node_size=10)
plt.savefig(dir_save_plots+"/posterior_predictive_simulated_household_network_"+pred_alg+"_"+sel_type+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_predictive_simulated_household_network_"+pred_alg+"_"+sel_type+".eps", bbox_inches='tight')
plt.show()


###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# Directory to save the simulations
# dir_posterior_pred = dir_base + "/posterior_pred_simu"
# if not os.path.exists(dir_posterior_pred):
#     os.makedirs(dir_posterior_pred)
#     print("Directory created")
# num_samples_post_pred = 50000
# list_G_sim_PP = []
# for i in range(num_samples_post_pred):
#     sampled_idx = random.randint(0, num_neigh-1)
#     sampled_weights = df_weights_RABC_obs.iloc[sampled_idx,:].to_list()
#     sampled_params = df_params_RABC_obs.iloc[sampled_idx,:].to_list()
#     G_sim = mixture_model_simulation(G_seed=G_seed, num_nodes=num_nodes,
#                                      weights=sampled_weights,
#                                      func_mechanisms=func_mechanisms,
#                                      args_mechanisms=[{'m':int(sampled_params[0]),'degree_aug':1},
#                                                       {'m':int(sampled_params[1])},
#                                                       {'m':int(sampled_params[2]),'degree_aug':1},
#                                                       {}]
#                                      )
#     G_sim_sums_sel = compute_indexed_summaries_undirected(G_sim, sel_sum_names)
#     list_G_sim_PP += [G_sim_sums_sel]
# df_G_sim_sums_sel_PP = pd.DataFrame(list_G_sim_PP)
# pickle.dump(df_G_sim_sums_sel_PP, open(dir_posterior_pred+"/posterior_predictive_"+str(num_samples_post_pred)+"_sim.p", "wb"))

num_samples_post_pred = 50000
df_G_sim_sums_sel_PP = pickle.load( open(resource_filename("mechanistic_net_abc", "data/4-household_analysis/SMC_recursive/posterior_predictive_"+str(num_samples_post_pred)+"_sim.p"), "rb"))

summaries_names = ["Number of edges", "Degree entropy", "Degree standard deviation",
                   "Transitivity", "Number of triangles", "Number of 2-cores",
                   "Number of 2-shells", "Number of shortest 5-paths",
                   "Number of shortest 6-paths", "Size minimum node dominating set"]

### Histograms of the summaries
for summary in range(df_G_sim_sums_sel_PP.shape[1]):
    plt.figure(figsize=(5,4))
    plt.hist(df_G_sim_sums_sel_PP.iloc[:,summary], color='tab:blue', alpha=0.5)
    plt.axvline(df_household_network_sums_sel.iloc[:,summary].values,
                color='tab:red', linestyle='dashed')
    # plt.title(df_summaries_cleanSmallRed_scen2.columns[summary])
    plt.xlabel(summaries_names[summary], size=13)
    plt.savefig(dir_save_plots+"/posterior_pred/marg_posterior_pred_"+df_household_network_sums_sel.columns[summary]+".pdf", bbox_inches='tight')
    plt.savefig(dir_save_plots+"/posterior_pred/marg_posterior_pred_"+df_household_network_sums_sel.columns[summary]+".eps", bbox_inches='tight')
    plt.show()

### Pairwise representation of the simulated summaries
for sum_i in range(df_G_sim_sums_sel_PP.shape[1]):
    for sum_j in range(df_G_sim_sums_sel_PP.shape[1]):
        if sum_i>sum_j:
            plt.figure(figsize=(5,4))
            plt.scatter(df_G_sim_sums_sel_PP.iloc[:,sum_i],
                        df_G_sim_sums_sel_PP.iloc[:,sum_j], c="tab:blue")
            plt.scatter(df_household_network_sums_sel.iloc[:,sum_i],
                        df_household_network_sums_sel.iloc[:,sum_j], c="tab:red")
            plt.xlabel(summaries_names[sum_i], size=13)
            plt.ylabel(summaries_names[sum_j], size=13)
            plt.savefig(dir_save_plots+"/posterior_pred/posterior_pred_"+df_household_network_sums_sel.columns[sum_i]+"_"+df_household_network_sums_sel.columns[sum_j]+".pdf", bbox_inches='tight')
            plt.savefig(dir_save_plots+"/posterior_pred/posterior_pred_"+df_household_network_sums_sel.columns[sum_i]+"_"+df_household_network_sums_sel.columns[sum_j]+".eps", bbox_inches='tight')
            plt.show()
