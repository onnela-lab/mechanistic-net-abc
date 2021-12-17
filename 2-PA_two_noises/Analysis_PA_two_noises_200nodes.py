# -*- coding: utf-8 -*-
"""
Second simulated example: PA to identify against two noise mechanisms.
"""

import networkx as nx
import pandas as pd
import numpy as np
import scipy.stats as ss
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle
import multiprocessing

from collections import Counter
from collections import defaultdict

from mechanistic_net_abc.summaries import compute_many_summaries_undirected
from mechanistic_net_abc.mechanisms import preferential_attachment_growth, random_attachment_growth, triangle_formation_node_addition
from mechanistic_net_abc.data_generation import data_ref_table_simulation
from mechanistic_net_abc.utility import drop_redundant_features
from mechanistic_net_abc.abc import abc_RSMCABC, distance_euclidean_std
from mechanistic_net_abc.summary_selection import recursiveElimination_RFMDA_select_summaries
from pkg_resources import resource_filename
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

### Base directory for this example on which results and plots will be saved
from mechanistic_net_abc.settings import base_dir_example2

#######################################################
### Define the general setting
#######################################################

example_number = "_example2"
num_cores = max(1,multiprocessing.cpu_count()-1)

### About the observed data
num_nodes = 200     # Number of observed nodes
m_true = 4          # True parameter for the preferential attachment
num_nodes_seed = 10 # Number of nodes in the seed graph (a fixed BA model with m_true)

### About the model of mixture
mech1 = preferential_attachment_growth
mech2 = random_attachment_growth
mech3 = triangle_formation_node_addition
func_mechanisms = [mech1, mech2, mech3]
num_mechs = len(func_mechanisms)

min_weights = [0,0,0]
max_weights = [1,1,1]

### About ABC
# We use as upper bound for the priors, the number of nodes in the seed network,
# a larger value would not be possible as the mechanism cannot create more edges
# to different nodes that there are in the seed graph.
max_m_value = num_nodes_seed

prior_m_pref_att = ss.randint(1, max_m_value+1)
prior_m_rand_att = ss.randint(1, max_m_value+1)
prior_args_mechanisms = [{'m':prior_m_pref_att}, {'m':prior_m_rand_att},{}]
fixed_args_mechanisms = [{'degree_aug':1},{},{}]
num_mech_params = sum([len(prior_args_mechanisms[i]) for i in range(len(prior_args_mechanisms))])

# Compute the prior standard deviation of the parameters (including weights)
std_prior_params = np.sqrt(ss.dirichlet([1]*num_mechs).var().tolist() + [prior_m_pref_att.var()] + [prior_m_rand_att.var()])

# Number of simulated data in the reference table
# (then splitted for summary selection, and inference)
num_sim = 100000
num_sim_sel_train = 40000
num_sim_sel_val = 10000
num_sim_knn = 50000

# Number of neighbors for K-NN-ABC:
num_neigh = 200

###!!! Set the base working directory, to change as you like from the settings.py file
dir_base = base_dir_example2
os.chdir(dir_base)

# Folder to save the reference table
dir_save_data = dir_base+"/data_"+str(num_nodes)+"nodes"+example_number
if not os.path.exists(dir_save_data):
    os.makedirs(dir_save_data)
    print("Directory created")

# Folder to save the recursive RF rankings
dir_save_rankings = dir_base+"/saved_rankings"
if not os.path.exists(dir_save_rankings):
    os.makedirs(dir_save_rankings)
    print("Directory created")

# Folder to save the figures
dir_save_plots = dir_base+"/saved_figures"
if not os.path.exists(dir_save_plots):
    os.makedirs(dir_save_plots)
    print("Directory created")

#######################################################
### Simulate three observed networks from PA mechanism
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

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper

# time1 = time.time()
# (df_weights_ref_table,
#   df_params_ref_table,
#   df_summaries_ref_table_init) = data_ref_table_simulation(G_seed = G_seed,
#                                                             num_sim = num_sim,
#                                                             num_nodes = num_nodes,
#                                                             func_mechanisms = func_mechanisms,
#                                                             prior_args_mechanisms = prior_args_mechanisms,
#                                                             fixed_args_mechanisms = fixed_args_mechanisms,
#                                                             num_cores = num_cores,
#                                                             min_weight = min_weights,
#                                                             max_weight = max_weights,
#                                                             many_summaries = True)
# time2 = time.time()
# print("Time to simulate the reference table: {} seconds.".format(time2 - time1))

# df_weights_ref_table.to_csv(dir_save_data+"/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
# df_params_ref_table.to_csv(dir_save_data+"/df_params_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)
# df_summaries_ref_table_init.to_csv(dir_save_data+"/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv", index=False)

df_weights_ref_table = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/ref_table/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv"))
df_params_ref_table = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/ref_table/df_params_ref_table_size_"+str(num_sim)+example_number+".csv"))
df_summaries_ref_table_init = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/ref_table/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv"))

###############################################################################
#### Drop the redundant features (i.e. no value variability or correlation 1)
###############################################################################

### We discard the summary statistics that present no variability

# For the reference table
df_summaries_ref_table_r = drop_redundant_features(df_summaries_ref_table_init)
# ['num_of_CC', 'num_nodes_LCC'] are removed because only one component

# For the observed summaries too
nunique = df_summaries_ref_table_init.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique==1].index
df_obs_sum_1_r = df_obs_sum_1_init.drop(cols_to_drop, axis=1)
df_obs_sum_2_r = df_obs_sum_2_init.drop(cols_to_drop, axis=1)
df_obs_sum_3_r = df_obs_sum_3_init.drop(cols_to_drop, axis=1)

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

# Drop response that are fixed if necessary (no prior on the parameter)
df_weights_params_ref_table = pd.concat([df_weights_ref_table,
                                         df_params_ref_table],
                                        axis=1, join="inner")

df_weights_params_ref_table = drop_redundant_features(df_weights_params_ref_table)

df_weights_params_scaled = df_weights_params_ref_table/std_prior_params


###############################################################################
### Split the reference table: one for summary selection (train + val.), one for ABC inference
###############################################################################

# For summary statistic selection (split in two, for selection and validation)
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
### Data visualization through PCA and PLS components
###############################################################################

### PCA analyses
pca_model = PCA(n_components = 4) # we want to project on the two first components
df_summaries_PCA = pca_model.fit_transform(df_summaries_scaled_sel_train)
df_obs_sum_1_PCA = pca_model.transform(df_obs_sum_1_scaled)
df_obs_sum_2_PCA = pca_model.transform(df_obs_sum_2_scaled)
df_obs_sum_3_PCA = pca_model.transform(df_obs_sum_3_scaled)

print("Percentage of variance explained: ", pca_model.explained_variance_ratio_)

# Plot the projected data with colors based on mechanism weights
fig, axis = plt.subplots(1, num_mechs, figsize=(14,7))
for k in range(num_mechs):
    tmp = axis[k].scatter(df_summaries_PCA[:,0],
                          df_summaries_PCA[:,1],
                          c=df_weights_params_sel_train.iloc[:,k], s=35, vmin=0, vmax=1)
    axis[k].scatter(df_obs_sum_1_PCA[:,0],
                    df_obs_sum_1_PCA[:,1],
                    c='tab:red', s=40)
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

PLSCA_model = PLSCanonical(n_components = 3)
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
                    c='tab:red', s=40)
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
### For the naive selection method: RF MDA + importance visualization
### optimize the m_try RF parameter based on out-of-bag score or 5-fold cross-val.
###############################################################################

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper

## With cross validation and mean squared error
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


# ### And with the out-of-bag error rate (mean squared error)
# oob_error_rates = []
# forest_reg_oob = RandomForestRegressor(n_estimators=500,
#                                           criterion='mse',
#                                           bootstrap=True,
#                                           oob_score=True,
#                                           n_jobs=num_cores,
#                                           random_state=123)

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
### Selection of summary stistics: train a RF (multi-output with scaled response)
###############################################################################

# Set the seed for reproducibility
rf_seed = 123

# m_try = m_try_CV
# m_try = m_try_oob
m_try = 25 # Selected by both CV and OOB error

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

mPA_truth_vs_pred_scaled = pd.DataFrame(data={'m_PA':response_all_sel_train[:,3], 'm_PA_hat':rf_model_unsel.oob_prediction_[:,3]*std_prior_params[3]})
mRA_truth_vs_pred_scaled = pd.DataFrame(data={'m_RA':response_all_sel_train[:,4], 'm_RA_hat':rf_model_unsel.oob_prediction_[:,4]*std_prior_params[4]})

sns.boxplot(x="m_PA", y="m_PA_hat", data=mPA_truth_vs_pred_scaled)
plt.show()
sns.boxplot(x="m_RA", y="m_RA_hat", data=mRA_truth_vs_pred_scaled)
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

mPA_truth_vs_pred_scaled_val = pd.DataFrame(data={'m_PA':response_all_sel_val[:,3], 'm_PA_hat':pred_rf_model_val[:,3]*std_prior_params[3]})
mRA_truth_vs_pred_scaled_val = pd.DataFrame(data={'m_RA':response_all_sel_val[:,4], 'm_RA_hat':pred_rf_model_val[:,4]*std_prior_params[4]})

sns.boxplot(x="m_PA", y="m_PA_hat", data=mPA_truth_vs_pred_scaled_val)
plt.show()
sns.boxplot(x="m_RA", y="m_RA_hat", data=mRA_truth_vs_pred_scaled_val)
plt.show()


# Predict the response of the three observed data
print(rf_model_unsel.predict(df_obs_sum_1_scaled)*std_prior_params)
print(rf_model_unsel.predict(df_obs_sum_2_scaled)*std_prior_params)
print(rf_model_unsel.predict(df_obs_sum_3_scaled)*std_prior_params)
# The discrete parameters are treated as continuous, thus the final predictions are numeric


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
num_features_to_plot = 40
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
# to avoid recomputing the importance:
# sorted_index_MDA = np.array([ 1, 30, 10, 39,  0, 18,  2, 34, 16,  3, 29,  8, 37, 28, 25, 27, 26, 33, 31, 24, 32, 23, 12, 22, 20, 19, 21, 17, 36,  9, 35, 38,  6, 14, 13,  5, 11,  7, 15,  4])
num_summaries_sel = 20
idx_summaries_rfMDA_sel = sorted_index_MDA[np.arange(-num_summaries_sel,0)]
print(name_summaries_unsel[idx_summaries_rfMDA_sel])

###############################################################################
### Perform summary statistic selection with recursive selection technique
###############################################################################

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper

n_estimators = 500
num_neigh_PCA = 100
num_neighbors_KNN = 200
n_repeats_MDA = 10
n_jobs = num_cores
pred_type = 'individual'

# For observation 1

obs_idx = 1

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
# print("Time Obs 1 recursif: ", time2 - time1)

# pickle.dump(average_RMSE_perResponse_recursif_obs1, open(dir_save_rankings+"/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(average_RMSE_total_recursif_obs1, open(dir_save_rankings+"/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(eliminated_features_recursif_obs1, open(dir_save_rankings+"/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(recursive_selected_summaries_obs1, open(dir_save_rankings+"/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))

average_RMSE_perResponse_recursif_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
average_RMSE_total_recursif_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
eliminated_features_recursif_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
recursive_selected_summaries_obs1 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))


# For observation 2

obs_idx = 2

# time1 = time.time()

# average_RMSE_perResponse_recursif_obs2, average_RMSE_total_recursif_obs2, eliminated_features_recursif_obs2, recursive_selected_summaries_obs2 =\
#     recursiveElimination_RFMDA_select_summaries(covariates_train=df_summaries_scaled_sel_train,
#                                                 responses_scaled_train=df_weights_params_scaled_sel_train,
#                                                 covariates_val=df_summaries_scaled_sel_val,
#                                                 responses_scaled_val=df_weights_params_scaled_sel_val,
#                                                 covariates_knn=df_summaries_scaled_knn,
#                                                 responses_scaled_knn=df_weights_params_scaled_knn,
#                                                 covariates_obs=df_obs_sum_2_scaled,
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
# print("Time Obs 2 recursif: ", time2 - time1)

# pickle.dump(average_RMSE_perResponse_recursif_obs2, open(dir_save_rankings+"/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(average_RMSE_total_recursif_obs2, open(dir_save_rankings+"/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(eliminated_features_recursif_obs2, open(dir_save_rankings+"/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(recursive_selected_summaries_obs2, open(dir_save_rankings+"/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))

average_RMSE_perResponse_recursif_obs2 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
average_RMSE_total_recursif_obs2 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
eliminated_features_recursif_obs2 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
recursive_selected_summaries_obs2 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))

# For observation 3

obs_idx = 3

# time1 = time.time()

# average_RMSE_perResponse_recursif_obs3, average_RMSE_total_recursif_obs3, eliminated_features_recursif_obs3, recursive_selected_summaries_obs3 =\
#     recursiveElimination_RFMDA_select_summaries(covariates_train=df_summaries_scaled_sel_train,
#                                                 responses_scaled_train=df_weights_params_scaled_sel_train,
#                                                 covariates_val=df_summaries_scaled_sel_val,
#                                                 responses_scaled_val=df_weights_params_scaled_sel_val,
#                                                 covariates_knn=df_summaries_scaled_knn,
#                                                 responses_scaled_knn=df_weights_params_scaled_knn,
#                                                 covariates_obs=df_obs_sum_3_scaled,
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
# print("Time Obs 3 recursif: ", time2 - time1)

# pickle.dump(average_RMSE_perResponse_recursif_obs3, open(dir_save_rankings+"/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(average_RMSE_total_recursif_obs3, open(dir_save_rankings+"/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(eliminated_features_recursif_obs3, open(dir_save_rankings+"/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))
# pickle.dump(recursive_selected_summaries_obs3, open(dir_save_rankings+"/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p", "wb"))

average_RMSE_perResponse_recursif_obs3 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/average_RMSE_perResponse_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
average_RMSE_total_recursif_obs3 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/average_RMSE_total_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
eliminated_features_recursif_obs3 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/eliminated_features_recursif_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))
recursive_selected_summaries_obs3 = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))


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
distances_obs_2, indices_obs_2 = nearest_neigh.kneighbors(df_obs_sum_2_scaled)
distances_obs_3, indices_obs_3 = nearest_neigh.kneighbors(df_obs_sum_3_scaled)

##### For the 1st observed network ######

obs_idx = 1

# Recover the weights
accepted_weights_obs_1_knn = df_weights_knn.iloc[indices_obs_1[0],:]

print("mean: ", np.array(accepted_weights_obs_1_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_1_knn.corr())
correla = accepted_weights_obs_1_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_1_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_1_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_1_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


# Recover the parameters
accepted_params_obs_1_knn = df_params_knn.iloc[indices_obs_1[0],:]

counter_tmp = accepted_params_obs_1_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_1_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


##### For the 2nd observed network ######

obs_idx = 2

# recover the accepted parameters, here the weights
accepted_weights_obs_2_knn = df_weights_knn.iloc[indices_obs_2[0],:]

print("mean: ", np.array(accepted_weights_obs_2_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_2_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_2_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_2_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_2_knn.corr())
correla = accepted_weights_obs_2_knn.corr()

# plot the posteriors
plt.hist(accepted_weights_obs_2_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_2_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_2_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_2_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_2_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_2_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()
sns.kdeplot(data=accepted_weights_obs_2_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


# recover the parameters
accepted_params_obs_2_knn = df_params_knn.iloc[indices_obs_2[0],:]

counter_tmp = accepted_params_obs_2_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_2_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_2_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


##### For the 3rd observed network ######

obs_idx = 3

# recover the accepted parameters, here the weights
accepted_weights_obs_3_knn = df_weights_knn.iloc[indices_obs_3[0],:]

print("mean: ", np.array(accepted_weights_obs_3_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_3_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_3_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_3_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_3_knn.corr())
correla = accepted_weights_obs_3_knn.corr()

# plot the posteriors
plt.hist(accepted_weights_obs_3_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_3_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_3_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_3_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_3_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_3_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_3_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# recover the parameters
accepted_params_obs_3_knn = df_params_knn.iloc[indices_obs_3[0],:]

counter_tmp = accepted_params_obs_3_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_3_knn.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_3_knn.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()



###############################################################################
###############!!! With naive summary statistic selection #####################

sel_type = "naive_RFMDA"
pred_alg = "knn"

idx_selection_from_rf = idx_summaries_rfMDA_sel
df_summaries_scaled_sel_knn = df_summaries_scaled_knn.iloc[:,idx_selection_from_rf]

df_obs_sum_1_scaled_sel_naive =  df_obs_sum_1_scaled.iloc[:,idx_selection_from_rf]
df_obs_sum_2_scaled_sel_naive =  df_obs_sum_2_scaled.iloc[:,idx_selection_from_rf]
df_obs_sum_3_scaled_sel_naive =  df_obs_sum_3_scaled.iloc[:,idx_selection_from_rf]

##### Estimate the parameters #####

nearest_neigh_sel = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
nearest_neigh_sel.fit(df_summaries_scaled_sel_knn)
distances_obs_1_sel, indices_obs_1_sel = nearest_neigh_sel.kneighbors(df_obs_sum_1_scaled_sel_naive)
distances_obs_2_sel, indices_obs_2_sel = nearest_neigh_sel.kneighbors(df_obs_sum_2_scaled_sel_naive)
distances_obs_3_sel, indices_obs_3_sel = nearest_neigh_sel.kneighbors(df_obs_sum_3_scaled_sel_naive)

##### For the 1st observed network ######

obs_idx = 1

# recover the accepted parameters, here the weights
accepted_weights_obs_1_naive_knn = df_weights_knn.iloc[indices_obs_1_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_1_naive_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_naive_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_naive_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_naive_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_1_naive_knn.corr())
correla = accepted_weights_obs_1_naive_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_1_naive_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_naive_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_1_naive_knn = df_params_knn.iloc[indices_obs_1_sel[0],:]

counter_tmp = accepted_params_obs_1_naive_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_naive_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_1_naive_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

##### For the 2nd observed network ######

obs_idx = 2

# recover the accepted parameters, here the weights
accepted_weights_obs_2_naive_knn = df_weights_knn.iloc[indices_obs_2_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_2_naive_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_2_naive_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_2_naive_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_2_naive_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_2_naive_knn.corr())
correla = accepted_weights_obs_2_naive_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_2_naive_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_2_naive_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_2_naive_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_2_naive_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_naive_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_naive_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_2_naive_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_2_naive_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_2_naive_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_2_naive_knn = df_params_knn.iloc[indices_obs_2_sel[0],:]

counter_tmp = accepted_params_obs_2_naive_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_2_naive_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_2_naive_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


##### For the 3rd observed network ######

obs_idx = 3

# recover the accepted parameters, here the weights
accepted_weights_obs_3_naive_knn = df_weights_knn.iloc[indices_obs_3_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_3_naive_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_3_naive_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_3_naive_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_3_naive_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_3_naive_knn.corr())
correla = accepted_weights_obs_3_naive_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_3_naive_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_3_naive_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_3_naive_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_3_naive_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_naive_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_naive_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_3_naive_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_3_naive_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_3_naive_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_3_naive_knn = df_params_knn.iloc[indices_obs_3_sel[0],:]

counter_tmp = accepted_params_obs_3_naive_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_3_naive_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_3_naive_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###############################################################################
###############!!! With recursive summary statistic selection #################

sel_type = "recursive_RFMDA"
pred_alg = "knn"


print("selected summaries obs1: ", recursive_selected_summaries_obs1)
print("number selected ", len(recursive_selected_summaries_obs1))

print("selected summaries obs2: ", recursive_selected_summaries_obs2)
print("number selected ", len(recursive_selected_summaries_obs2))

print("selected summaries obs3: ", recursive_selected_summaries_obs3)
print("number selected ", len(recursive_selected_summaries_obs3))


# Reduce the reference table

df_summaries_scaled_sel_knn_obs1 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs1]
df_summaries_scaled_sel_knn_obs2 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs2]
df_summaries_scaled_sel_knn_obs3 = df_summaries_scaled_knn.iloc[:,recursive_selected_summaries_obs3]

df_obs_sum_1_scaled_sel =  df_obs_sum_1_scaled.iloc[:,recursive_selected_summaries_obs1]
df_obs_sum_2_scaled_sel =  df_obs_sum_2_scaled.iloc[:,recursive_selected_summaries_obs2]
df_obs_sum_3_scaled_sel =  df_obs_sum_3_scaled.iloc[:,recursive_selected_summaries_obs3]

##### Represent the summary statistics values based on weight/parameter values to check their informativeness

### Bivariate distribution of the summary statistics

for sum_i in range(df_summaries_scaled_sel_knn_obs1.shape[1]):
    for sum_j in range(df_summaries_scaled_sel_knn_obs1.shape[1]):
        if sum_i>sum_j:
            fig, axis = plt.subplots(1, num_mechs, figsize=(14,7))
            for k in range(num_mechs):
                tmp = axis[k].scatter(df_summaries_scaled_sel_knn_obs1.iloc[:,sum_i],
                                      df_summaries_scaled_sel_knn_obs1.iloc[:,sum_j],
                                      c=df_weights_params_knn.iloc[:,k], s=35,
                                      vmin=0, vmax=1)
                axis[k].scatter(df_obs_sum_1_scaled_sel.iloc[:,sum_i],
                                df_obs_sum_1_scaled_sel.iloc[:,sum_j],
                                label=r"$G_1^*$", c='tab:red')
                axis[k].scatter(df_obs_sum_2_scaled_sel.iloc[:,sum_i],
                                df_obs_sum_2_scaled_sel.iloc[:,sum_j],
                                label=r"$G_2^*$", c='magenta')
                axis[k].scatter(df_obs_sum_3_scaled_sel.iloc[:,sum_i],
                                df_obs_sum_3_scaled_sel.iloc[:,sum_j],
                               label=r"$G_3^*$", c='cyan')
                fig.colorbar(tmp, ax=axis[k])
                if k ==0:
                    axis[k].set_ylabel(df_summaries_scaled_sel_knn_obs1.columns[sum_j])
                axis[k].set_xlabel(df_summaries_scaled_sel_knn_obs1.columns[sum_i])
                # fig.legend()
            fig.show()


for sum_i in range(df_summaries_scaled_sel_knn_obs1.shape[1]):
    for sum_j in range(df_summaries_scaled_sel_knn_obs1.shape[1]):
        if sum_i>sum_j:
            fig, axis = plt.subplots(1, 2, figsize=(14,7))
            kk=0
            for k in [3,4]:
                tmp = axis[kk].scatter(df_summaries_scaled_sel_knn_obs1.iloc[:,sum_i],
                                       df_summaries_scaled_sel_knn_obs1.iloc[:,sum_j],
                                       c=df_weights_params_knn.iloc[:,k], s=35,
                                       vmin=1, vmax=10)
                axis[kk].scatter(df_obs_sum_1_scaled_sel.iloc[:,sum_i],
                                df_obs_sum_1_scaled_sel.iloc[:,sum_j],
                                label=r"$G_1^*$", c='tab:red')
                axis[kk].scatter(df_obs_sum_2_scaled_sel.iloc[:,sum_i],
                                df_obs_sum_2_scaled_sel.iloc[:,sum_j],
                                label=r"$G_2^*$", c='magenta')
                axis[kk].scatter(df_obs_sum_3_scaled_sel.iloc[:,sum_i],
                                df_obs_sum_3_scaled_sel.iloc[:,sum_j],
                               label=r"$G_3^*$", c='cyan')
                fig.colorbar(tmp, ax=axis[kk])
                if kk==0:
                    axis[kk].set_ylabel(df_summaries_scaled_sel_knn_obs1.columns[sum_j])
                axis[kk].set_xlabel(df_summaries_scaled_sel_knn_obs1.columns[sum_i])
                kk += 1
                # fig.legend()
            fig.show()

### Univariate distribution of the summary statistics

randunif = ss.uniform.rvs(size=df_summaries_scaled_sel_knn_obs1.shape[0])
for sum_i in range(df_summaries_scaled_sel_knn_obs1.shape[1]):
    fig, axis = plt.subplots(1, num_mechs, figsize=(14,7))
    for k in range(num_mechs):
        tmp =  axis[k].scatter(df_summaries_scaled_sel_knn_obs1.iloc[:,sum_i],
                               randunif,
                               c=df_weights_params_knn.iloc[:,k], s=35,
                               vmin=0, vmax=1)
        axis[k].scatter(df_obs_sum_1_scaled_sel.iloc[:,sum_i],
                        0,
                        label=r"$G_1^*$", c='tab:red')
        axis[k].scatter(df_obs_sum_2_scaled_sel.iloc[:,sum_i],
                        0,
                        label=r"$G_2^*$", c='magenta')
        axis[k].scatter(df_obs_sum_3_scaled_sel.iloc[:,sum_i],
                        0,
                        label=r"$G_3^*$", c='cyan')
        fig.colorbar(tmp, ax=axis[k])
        if k ==0:
            axis[k].set_ylabel("Unif noise")
        axis[k].set_xlabel(df_summaries_scaled_sel_knn_obs1.columns[sum_i])
        # fig.legend()
    fig.show()

for sum_i in range(df_summaries_scaled_sel_knn_obs1.shape[1]):
    fig, axis = plt.subplots(1, 2, figsize=(14,7))
    kk = 0
    for k in [3,4]:
        tmp =  axis[kk].scatter(df_summaries_scaled_sel_knn_obs1.iloc[:,sum_i],
                                randunif,
                                c=df_weights_params_knn.iloc[:,k], s=35,
                                vmin=1, vmax=10)
        axis[kk].scatter(df_obs_sum_1_scaled_sel.iloc[:,sum_i],
                         0,
                         label=r"$G_1^*$", c='tab:red')
        axis[kk].scatter(df_obs_sum_2_scaled_sel.iloc[:,sum_i],
                         0,
                         label=r"$G_2^*$", c='magenta')
        axis[kk].scatter(df_obs_sum_3_scaled_sel.iloc[:,sum_i],
                         0,
                         label=r"$G_3^*$", c='cyan')
        fig.colorbar(tmp, ax=axis[kk])
        if kk ==0:
            axis[kk].set_ylabel("Unif noise")
        axis[kk].set_xlabel(df_summaries_scaled_sel_knn_obs1.columns[sum_i])
        # fig.legend()
        kk += 1
    fig.show()

##### Interesting summary representation that highlight which summary is relevant for which parameter

sample_size = 15000
randunif = ss.uniform.rvs(size=sample_size)

### Summary selected and relevant for TF weight is average_clustering_coefficient
fig, axis = plt.subplots(1, 1, figsize=(5,5))
tmp =  axis.scatter(df_summaries_scaled_sel_knn_obs1['avg_clustering_coef'][:sample_size],
                    randunif,
                    c=df_weights_params_knn.iloc[:sample_size,2], s=35,
                    vmin=0, vmax=1)
axis.axvline(x=df_obs_sum_1_scaled_sel['avg_clustering_coef'][0], linestyle='dashed', c='tab:red')
axis.axvline(x=df_obs_sum_2_scaled_sel['avg_clustering_coef'][0], linestyle='dashed', c='magenta')
axis.axvline(x=df_obs_sum_3_scaled_sel['avg_clustering_coef'][0], linestyle='dashed', c='cyan')
cbr = fig.colorbar(tmp, ax=axis)
cbr.ax.set_title(r"$\alpha_{TF}$", {'fontsize': 15})
axis.set_ylabel("Uniform noise", size=13)
axis.set_xlabel("Average clustering coefficient", size=13)
fig.savefig(dir_save_plots+"/summaries_avg_clust_coef_noise_wTF_"+sel_type+example_number+".pdf", bbox_inches='tight')
fig.savefig(dir_save_plots+"/summaries_avg_clust_coef_noise_wTF_"+sel_type+example_number+".eps", bbox_inches='tight')
fig.show()

### Degree standard deviation seems relevant for m_PA
fig, axis = plt.subplots(1, 1, figsize=(5,5))
tmp =  axis.scatter(df_summaries_scaled_sel_knn_obs1['degree_std'][:sample_size],
                    randunif,
                    c=df_weights_params_knn.iloc[:sample_size,3], s=35,
                    vmin=1, vmax=10)
axis.axvline(x=df_obs_sum_1_scaled_sel['degree_std'][0], linestyle='dashed', c='tab:red')
axis.axvline(x=df_obs_sum_2_scaled_sel['degree_std'][0], linestyle='dashed', c='magenta')
axis.axvline(x=df_obs_sum_3_scaled_sel['degree_std'][0], linestyle='dashed', c='cyan')
cbr = fig.colorbar(tmp, ax=axis)
cbr.ax.set_title(r"$m_{PA}$", {'fontsize': 15})
axis.set_ylabel("Uniform noise", size=13)
axis.set_xlabel("Degree standard deviation", size=13)
fig.savefig(dir_save_plots+"/summaries_deg_std_noise_mPA_"+sel_type+example_number+".pdf", bbox_inches='tight')
fig.savefig(dir_save_plots+"/summaries_deg_std_noise_mPA_"+sel_type+example_number+".eps", bbox_inches='tight')
fig.show()

### Degree entropy seems relevant for w_RA for large values at least
fig, axis = plt.subplots(1, 1, figsize=(5,5))
tmp =  axis.scatter(df_summaries_scaled_sel_knn_obs1['degree_entropy'][:sample_size],
                    randunif,
                    c=df_weights_params_knn.iloc[:sample_size,1], s=35,
                    vmin=0, vmax=1)
axis.axvline(x=df_obs_sum_1_scaled_sel['degree_entropy'][0], linestyle='dashed', c='tab:red')
axis.axvline(x=df_obs_sum_2_scaled_sel['degree_entropy'][0], linestyle='dashed', c='magenta')
axis.axvline(x=df_obs_sum_3_scaled_sel['degree_entropy'][0], linestyle='dashed', c='cyan')
cbr = fig.colorbar(tmp, ax=axis)
cbr.ax.set_title(r"$\alpha_{RA}$", {'fontsize': 15})
axis.set_ylabel("Uniform noise", size=13)
axis.set_xlabel("Degree entropy", size=13)
fig.savefig(dir_save_plots+"/summaries_deg_entropy_noise_wRA_"+sel_type+example_number+".pdf", bbox_inches='tight')
fig.savefig(dir_save_plots+"/summaries_deg_entropy_noise_wRA_"+sel_type+example_number+".eps", bbox_inches='tight')
fig.show()

### Jointly relevant for TF weight

fig, axis = plt.subplots(1, 1, figsize=(5,5))
tmp =  axis.scatter(df_summaries_scaled_sel_knn_obs1['num_4cores'][:sample_size],
                    df_summaries_scaled_sel_knn_obs1['transitivity'][:sample_size],
                    c=df_weights_params_knn.iloc[:sample_size,2], s=35,
                    vmin=0, vmax=1)
axis.scatter(df_obs_sum_1_scaled_sel['num_4cores'],
             df_obs_sum_1_scaled_sel['transitivity'],
             linestyle='dashed', c='tab:red')
axis.scatter(df_obs_sum_2_scaled_sel['num_4cores'],
             df_obs_sum_2_scaled_sel['transitivity'],
             linestyle='dashed', c='magenta')
axis.scatter(df_obs_sum_3_scaled_sel['num_4cores'],
             df_obs_sum_3_scaled_sel['transitivity'],
             linestyle='dashed', c='cyan')
cbr = fig.colorbar(tmp, ax=axis)
cbr.ax.set_title(r"$\alpha_{TF}$", {'fontsize': 15})
axis.set_ylabel("Transitivity", size=13)
axis.set_xlabel("Number of 4-cores", size=13)
fig.savefig(dir_save_plots+"/summaries_4cores_transitivity_wTF_"+sel_type+example_number+".pdf", bbox_inches='tight')
fig.savefig(dir_save_plots+"/summaries_4cores_transitivity_wTF_"+sel_type+example_number+".eps", bbox_inches='tight')
fig.show()

fig, axis = plt.subplots(1, 1, figsize=(5,5))
tmp =  axis.scatter(df_summaries_scaled_sel_knn_obs1['num_4cores'][:sample_size],
                    df_summaries_scaled_sel_knn_obs1['degree_std'][:sample_size],
                    c=df_weights_params_knn.iloc[:sample_size,2], s=35,
                    vmin=0, vmax=1)
axis.scatter(df_obs_sum_1_scaled_sel['num_4cores'],
             df_obs_sum_1_scaled_sel['degree_std'],
             linestyle='dashed', c='tab:red')
axis.scatter(df_obs_sum_2_scaled_sel['num_4cores'],
             df_obs_sum_2_scaled_sel['degree_std'],
             linestyle='dashed', c='magenta')
axis.scatter(df_obs_sum_3_scaled_sel['num_4cores'],
             df_obs_sum_3_scaled_sel['degree_std'],
             linestyle='dashed', c='cyan')
cbr = fig.colorbar(tmp, ax=axis)
cbr.ax.set_title(r"$\alpha_{TF}$", {'fontsize': 15})
axis.set_ylabel("Degree standard deviation", size=13)
axis.set_xlabel("Number of 4-cores", size=13)
fig.savefig(dir_save_plots+"/summaries_4cores_deg_std_wTF_"+sel_type+example_number+".pdf", bbox_inches='tight')
fig.savefig(dir_save_plots+"/summaries_4cores_deg_std_wTF_"+sel_type+example_number+".eps", bbox_inches='tight')
fig.show()

### Interesting for w_PA

fig, axis = plt.subplots(1, 1, figsize=(5,5))
tmp =  axis.scatter(df_summaries_scaled_sel_knn_obs1['transitivity'][:sample_size],
                    df_summaries_scaled_sel_knn_obs1['degree_entropy'][:sample_size],
                    c=df_weights_params_knn.iloc[:sample_size,0], s=35,
                    vmin=0, vmax=1)
axis.scatter(df_obs_sum_1_scaled_sel['transitivity'],
             df_obs_sum_1_scaled_sel['degree_entropy'],
             linestyle='dashed', c='tab:red')
axis.scatter(df_obs_sum_2_scaled_sel['transitivity'],
             df_obs_sum_2_scaled_sel['degree_entropy'],
             linestyle='dashed', c='magenta')
axis.scatter(df_obs_sum_3_scaled_sel['transitivity'],
             df_obs_sum_3_scaled_sel['degree_entropy'],
             linestyle='dashed', c='cyan')
cbr = fig.colorbar(tmp, ax=axis)
cbr.ax.set_title(r"$\alpha_{PA}$", {'fontsize': 15})
axis.set_ylabel("Degree entropy", size=13)
axis.set_xlabel("Transitivity", size=13)
fig.savefig(dir_save_plots+"/summaries_transitivity_deg_entropy_wPA_"+sel_type+example_number+".pdf", bbox_inches='tight')
fig.savefig(dir_save_plots+"/summaries_transitivity_deg_entropy_wPA_"+sel_type+example_number+".eps", bbox_inches='tight')
fig.show()


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

obs_idx = 1

# recover the accepted parameters, here the weights
accepted_weights_obs_1_sel_knn = df_weights_knn.iloc[indices_obs_1_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_1_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_1_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_1_sel_knn.corr())
correla = accepted_weights_obs_1_sel_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_1_sel_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_1_sel_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_1_sel_knn = df_params_knn.iloc[indices_obs_1_sel[0],:]

counter_tmp = accepted_params_obs_1_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_1_sel_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_1_sel_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

##### For the 2nd observed network ######

obs_idx = 2

# recover the accepted parameters, here the weights
accepted_weights_obs_2_sel_knn = df_weights_knn.iloc[indices_obs_2_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_2_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_2_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_2_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_2_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_2_sel_knn.corr())
correla = accepted_weights_obs_2_sel_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_2_sel_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_2_sel_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_sel_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_sel_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_2_sel_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_2_sel_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_2_sel_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_2_sel_knn = df_params_knn.iloc[indices_obs_2_sel[0],:]

counter_tmp = accepted_params_obs_2_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_2_sel_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_2_sel_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


##### For the 3rd observed network ######

obs_idx = 3

# recover the accepted parameters, here the weights
accepted_weights_obs_3_sel_knn = df_weights_knn.iloc[indices_obs_3_sel[0],:]

print("mean: ", np.array(accepted_weights_obs_3_sel_knn.apply(np.mean)))
print("median: ", np.array(accepted_weights_obs_3_sel_knn.apply(np.median)))

print("95% credible intervals: ", accepted_weights_obs_3_sel_knn.apply(np.quantile, q=(0.025, 0.975)))
print("50% credible intervals: ", accepted_weights_obs_3_sel_knn.apply(np.quantile, q=(0.25, 0.75)))

print(accepted_weights_obs_3_sel_knn.corr())
correla = accepted_weights_obs_3_sel_knn.corr()

# Plot the posteriors
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(accepted_weights_obs_3_sel_knn.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=accepted_weights_obs_3_sel_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_sel_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_sel_knn, x="weight_mech_3", kde=True,
             color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=accepted_weights_obs_3_sel_knn, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_3_sel_knn, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=accepted_weights_obs_3_sel_knn, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_3_sel_knn = df_params_knn.iloc[indices_obs_3_sel[0],:]

counter_tmp = accepted_params_obs_3_sel_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.hist(accepted_params_obs_3_sel_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_3_sel_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()



############################## SMC-ABC ########################################

###############################################################################
### Run SMC-ABC without summary statistic selection
###############################################################################

sel_type="unselected_summaries"
pred_alg = "SMC"

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
sel_type = "unselected_summaries"
###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# dir_save_results = dir_base+"/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
# if not os.path.exists(dir_save_results):
#     os.makedirs(dir_save_results)
#     print("Directory created")

###### For the first observed data ######
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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the second observed data ######
obs_idx = 2
df_obs_sum_for_RABC = df_obs_sum_2

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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the first observed data ######
obs_idx = 3
df_obs_sum_for_RABC = df_obs_sum_3

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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###############################################################################
###!!! Run the SMC-ABC algorithm on the summary statistics selected naively
###############################################################################

sel_type = "naive_RFMDA"
pred_alg = "SMC"
idx_selection_from_rf = idx_summaries_rfMDA_sel

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# Set the directory to same the SMC-ABC results
# dir_save_results = dir_base+"/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
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

###### For the first observed data ######
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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the second observed data ######
obs_idx = 2
df_obs_sum_for_RABC = df_obs_sum_2

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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the first observed data ######
obs_idx = 3
df_obs_sum_for_RABC = df_obs_sum_3

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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_naive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###############################################################################
###!!! Run the SMC-ABC algorithm on the summary statistics selected with recursive method
###############################################################################

sel_type = "recursive_RFMDA"
pred_alg = "SMC"

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper
# Set the directory to same the SMC-ABC results
# dir_save_results = dir_base+"/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
# if not os.path.exists(dir_save_results):
#     os.makedirs(dir_save_results)
#     print("Directory created")

###!!! Uncomment to rerun, preferably on a cluster, else use the results from our paper

###### For the first observed data ######

# Load the selected summaries with recursive method
obs_idx = 1
recursive_selected_summaries_obs = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))

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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()



###### For the second observed data ######

# Load the selected summaries with recursive method
obs_idx = 2
recursive_selected_summaries_obs = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))

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


df_obs_sum_for_RABC = df_obs_sum_2[sel_sum_names]


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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the Third observed data ######

# Load the selected summaries with recursive method
obs_idx = 3
recursive_selected_summaries_obs = pickle.load(open(resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/rankings/recursive_selected_summaries_obs"+str(obs_idx)+"_"+pred_type+example_number+".p"), "rb"))

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


df_obs_sum_for_RABC = df_obs_sum_3[sel_sum_names]


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

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_weights_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_params_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/2-PA_two_noises/SMC_recursive/df_dist_acc_RABC_obs"+str(obs_idx)+"_"+sel_type+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())
correla = df_weights_RABC_obs.corr()

# plot the posteriors
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.hist(df_weights_RABC_obs.iloc[:,2], alpha=0.5, label=r'$\alpha_{TF}$')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_3", kde=True,
              color='tab:green', element="bars", label=r'$\alpha_{TF}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=13)
plt.ylabel("Counts", size=13)
plt.legend(loc='upper center', prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Joint posteriors
sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_2",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,1],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{RA}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_RA_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_1", y="weight_mech_3",
            color='tab:blue')
plt.plot(1,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[0,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{PA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wPA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

sns.kdeplot(data=df_weights_RABC_obs, x="weight_mech_2", y="weight_mech_3",
            color='tab:blue')
plt.plot(0,0, 'ro')
plt.text(0.85, 1, "corr.: "+str(round(correla.iloc[1,2],3)), fontsize=13)
plt.xlabel(r'$\alpha_{RA}$', size=13)
plt.ylabel(r'$\alpha_{TF}$', size=13)
plt.xlim(-0.25,1.25)
plt.ylim(-0.25,1.25)
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_joint_weights_wRA_TF_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=13)
plt.ylabel("Counts", size=13)
plt.legend(prop={'size': 13})
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_"+pred_alg+"_"+sel_type+"_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()