# -*- coding: utf-8 -*-
"""
In this example, we generate Barabasi-Albert networks, i.e. the only underlying
mechanism is the preferential attachment (PA) mechanism, and we want to identify that
this mechanism dominates thanks to a model of mixture of mechanisms.
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
import multiprocessing
import os
import seaborn as sns
from collections import Counter
from collections import defaultdict
from mechanistic_net_abc.summaries import compute_many_summaries_undirected
from mechanistic_net_abc.mechanisms import preferential_attachment_growth, random_attachment_growth
from mechanistic_net_abc.data_generation import data_ref_table_simulation
from mechanistic_net_abc.utility import drop_redundant_features
from mechanistic_net_abc.abc import abc_RSMCABC, distance_euclidean_std
from pkg_resources import resource_filename
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical

### Base directory for this example on which results and plots will be saved
from mechanistic_net_abc.settings import base_dir_example1

#######################################################
### Define the general simulation setting
#######################################################

example_number = "_example1"
num_cores = max(1,multiprocessing.cpu_count()-1)

### About the observed data
num_nodes = 200     # Number of observed nodes
m_true = 4          # True parameter for the preferential attachment
num_nodes_seed = 10 # Number of nodes in the seed graph (a fixed BA model with m_true)

### About the model of mixture
mech1 = preferential_attachment_growth
mech2 = random_attachment_growth
func_mechanisms = [mech1, mech2]
num_mechs = len(func_mechanisms)

min_weights = [0,0]
max_weights = [1,1]

### About ABC
# We use as upper bound for the priors, the number of nodes in the seed network,
# a larger value would not be possible as the mechanism cannot create more edges
# to different nodes that there are in the seed graph.
max_m_value = num_nodes_seed

prior_m_pref_att = ss.randint(1, max_m_value+1)
prior_m_rand_att = ss.randint(1, max_m_value+1)
prior_args_mechanisms = [{'m':prior_m_pref_att}, {'m':prior_m_rand_att}]
fixed_args_mechanisms = [{'degree_aug':1}, {}]
num_mech_params = num_mech_params = sum([len(prior_args_mechanisms[i]) for i in range(len(prior_args_mechanisms))])

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
dir_base = base_dir_example1
os.chdir(dir_base)

### Directory to save plots
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

###!!! We suggest running the simulation of the reference table on a cluster

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

# Use the data used in our paper
df_weights_ref_table = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/ref_table/df_weights_ref_table_size_"+str(num_sim)+example_number+".csv"))
df_params_ref_table = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/ref_table/df_params_ref_table_size_"+str(num_sim)+example_number+".csv"))
df_summaries_ref_table_init = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/ref_table/df_summaries_ref_table_size_"+str(num_sim)+example_number+".csv"))

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
pca_model = PCA(n_components = 4)
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
###!!! Perform inference with K-NN-ABC algorithm
###############################################################################

############### Without summary statistic selection ###########################

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

# Plot the posteriors
plt.figure(figsize=(5, 5))
plt.hist(accepted_weights_obs_1_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$', edgecolor=None)
plt.hist(accepted_weights_obs_1_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$', edgecolor=None)
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
plt.figure(figsize=(5, 5))
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_1_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_density_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# Recover the parameters
accepted_params_obs_1_knn = df_params_knn.iloc[indices_obs_1[0],:]

counter_tmp = accepted_params_obs_1_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.figure(figsize=(5, 5))
plt.hist(accepted_params_obs_1_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_1_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_params_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
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

# plot the posteriors
plt.figure(figsize=(5, 5))
plt.hist(accepted_weights_obs_2_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_2_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
plt.figure(figsize=(5, 5))
sns.histplot(data=accepted_weights_obs_2_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_2_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_density_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# recover the parameters
accepted_params_obs_2_knn = df_params_knn.iloc[indices_obs_2[0],:]

counter_tmp = accepted_params_obs_2_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.figure(figsize=(5, 5))
plt.hist(accepted_params_obs_2_knn.iloc[:,0], alpha=0.5, label='$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_2_knn.iloc[:,1], alpha=0.5, label='$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_params_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
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

# plot the posteriors
plt.figure(figsize=(5, 5))
plt.hist(accepted_weights_obs_3_knn.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(accepted_weights_obs_3_knn.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
plt.figure(figsize=(5, 5))
sns.histplot(data=accepted_weights_obs_3_knn, x="weight_mech_1", kde=True,
             color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=accepted_weights_obs_3_knn, x="weight_mech_2", kde=True,
             color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_density_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# recover the parameters
accepted_params_obs_3_knn = df_params_knn.iloc[indices_obs_3[0],:]

counter_tmp = accepted_params_obs_3_knn.apply(Counter)
for i in range(len(counter_tmp)):
    print(counter_tmp[i].most_common(1))

plt.figure(figsize=(5, 5))
plt.hist(accepted_params_obs_3_knn.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(accepted_params_obs_3_knn.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_params_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_knn_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###############################################################################
### Run SMC-ABC without summary statistic selection
###############################################################################

###!!! We suggest running this part on a cluster

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
dir_save_results = dir_base+"/"+sel_type+"/results_"+str(num_nodes)+"nodes_SMC"+example_number
if not os.path.exists(dir_save_results):
    os.makedirs(dir_save_results)
    print("Directory created")


###### For the first observed data ######
obs_idx = 1
df_obs_sum_for_RABC = df_obs_sum_1

# time1 = time.time()
# (df_weights_RABC_obs,
#  df_params_RABC_obs,
#  df_dist_acc_RABC_obs,
#  sim_count_total_obs,
#  threshold_values_obs) = abc_RSMCABC(G_seed=G_seed,
#                                      num_nodes=num_nodes,
#                                      func_mechanisms=func_mechanisms,
#                                      prior_args_mechanisms=prior_args_mechanisms,
#                                      fixed_args_mechanisms=fixed_args_mechanisms,
#                                      min_weight=min_weights,
#                                      max_weight=max_weights,
#                                      threshold_init=threshold_init,
#                                      threshold_final=threshold_final,
#                                      alpha=alpha,
#                                      scale_factor=scale_factor,
#                                      weight_perturbation=weight_perturbation,
#                                      num_acc_sim=num_acc_sim,
#                                      df_observed_summaries=df_obs_sum_for_RABC,
#                                      distance_func=distance_func,
#                                      distance_args=distance_args,
#                                      sel_sum_names=sel_sum_names,
#                                      max_sim=max_sim)

# time2 = time.time()
# print("Time ABC-RSMC: ", time2 - time1)
# print("Total number of simulations: ", sim_count_total_obs)

# df_weights_RABC_obs.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)
# df_params_RABC_obs.to_csv(dir_save_results+"/df_params_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)
# df_dist_acc_RABC_obs.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())

# plot the posteriors
plt.figure(figsize=(5, 5))
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
plt.figure(figsize=(5, 5))
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_density_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.figure(figsize=(5, 5))
plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_params_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the second observed data ######
obs_idx = 2
df_obs_sum_for_RABC = df_obs_sum_2

# time1 = time.time()
# (df_weights_RABC_obs,
#  df_params_RABC_obs,
#  df_dist_acc_RABC_obs,
#  sim_count_total_obs,
#  threshold_values_obs) = abc_RSMCABC(G_seed=G_seed,
#                                      num_nodes=num_nodes,
#                                      func_mechanisms=func_mechanisms,
#                                      prior_args_mechanisms=prior_args_mechanisms,
#                                      fixed_args_mechanisms=fixed_args_mechanisms,
#                                      min_weight=min_weights,
#                                      max_weight=max_weights,
#                                      threshold_init=threshold_init,
#                                      threshold_final=threshold_final,
#                                      alpha=alpha,
#                                      scale_factor=scale_factor,
#                                      weight_perturbation=weight_perturbation,
#                                      num_acc_sim=num_acc_sim,
#                                      df_observed_summaries=df_obs_sum_for_RABC,
#                                      distance_func=distance_func,
#                                      distance_args=distance_args,
#                                      sel_sum_names=sel_sum_names,
#                                      max_sim=max_sim)

# time2 = time.time()
# print("Time ABC-RSMC: ", time2 - time1)
# print("Total number of simulations: ", sim_count_total_obs)

# df_weights_RABC_obs.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)
# df_params_RABC_obs.to_csv(dir_save_results+"/df_params_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)
# df_dist_acc_RABC_obs.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())

# plot the posteriors
plt.figure(figsize=(5, 5))
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
plt.figure(figsize=(5, 5))
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_density_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.figure(figsize=(5, 5))
plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_params_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()


###### For the first observed data ######
obs_idx = 3
df_obs_sum_for_RABC = df_obs_sum_3

# time1 = time.time()
# (df_weights_RABC_obs,
#  df_params_RABC_obs,
#  df_dist_acc_RABC_obs,
#  sim_count_total_obs,
#  threshold_values_obs) = abc_RSMCABC(G_seed=G_seed,
#                                      num_nodes=num_nodes,
#                                      func_mechanisms=func_mechanisms,
#                                      prior_args_mechanisms=prior_args_mechanisms,
#                                      fixed_args_mechanisms=fixed_args_mechanisms,
#                                      min_weight=min_weights,
#                                      max_weight=max_weights,
#                                      threshold_init=threshold_init,
#                                      threshold_final=threshold_final,
#                                      alpha=alpha,
#                                      scale_factor=scale_factor,
#                                      weight_perturbation=weight_perturbation,
#                                      num_acc_sim=num_acc_sim,
#                                      df_observed_summaries=df_obs_sum_for_RABC,
#                                      distance_func=distance_func,
#                                      distance_args=distance_args,
#                                      sel_sum_names=sel_sum_names,
#                                      max_sim=max_sim)

# time2 = time.time()
# print("Time ABC-RSMC: ", time2 - time1)
# print("Total number of simulations: ", sim_count_total_obs)

# df_weights_RABC_obs.to_csv(dir_save_results+"/df_weights_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)
# df_params_RABC_obs.to_csv(dir_save_results+"/df_params_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)
# df_dist_acc_RABC_obs.to_csv(dir_save_results+"/df_dist_acc_RABC_obs"+str(obs_idx)+example_number+".csv", index=False)

df_weights_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_weights_RABC_obs"+str(obs_idx)+example_number+".csv"))
df_params_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_params_RABC_obs"+str(obs_idx)+example_number+".csv"))
df_dist_acc_RABC_obs = pd.read_csv(filepath_or_buffer = resource_filename("mechanistic_net_abc", "data/1-PA_one_noise/SMC_unselected/df_dist_acc_RABC_obs"+str(obs_idx)+example_number+".csv"))

### Plot of the posteriors ###

# For the weights
print(np.array(df_weights_RABC_obs.apply(np.mean)))
print(np.array(df_weights_RABC_obs.apply(np.median)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.025, 0.975)))

print(df_weights_RABC_obs.apply(np.quantile, q=(0.25, 0.75)))

print(df_weights_RABC_obs.corr())

# plot the posteriors
plt.figure(figsize=(5, 5))
plt.hist(df_weights_RABC_obs.iloc[:,0], alpha=0.5, label=r'$\alpha_{PA}$')
plt.hist(df_weights_RABC_obs.iloc[:,1], alpha=0.5, label=r'$\alpha_{RA}$')
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# with seaborn
plt.figure(figsize=(5, 5))
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_1", kde=True,
              color='tab:blue', element="bars", label=r'$\alpha_{PA}$', edgecolor=None)
sns.histplot(data=df_weights_RABC_obs, x="weight_mech_2", kde=True,
              color='tab:orange', element="bars", label=r'$\alpha_{RA}$', edgecolor=None)
plt.legend(loc='upper center')
plt.xlabel("Mechanism weights", size=15)
plt.ylabel("Counts", size=15)
plt.legend(loc='upper center', prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_weights_density_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_weights_density_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()

# For the mechanism parameters
tmp_counter = df_params_RABC_obs.apply(Counter)
for i in range(len(tmp_counter)):
    print(tmp_counter[i].most_common(1))

plt.figure(figsize=(5, 5))
plt.hist(df_params_RABC_obs.iloc[:,0], alpha=0.5, label=r'$m_{PA}$', bins=np.arange(0.5,11.5))
plt.hist(df_params_RABC_obs.iloc[:,1], alpha=0.5, label=r'$m_{RA}$', bins=np.arange(0.5,11.5))
plt.xlabel("Mechanism parameters", size=15)
plt.ylabel("Counts", size=15)
plt.legend(prop={'size': 15})
plt.savefig(dir_save_plots+"/posterior_params_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/posterior_params_SMC_unselected_summaries_obs"+str(obs_idx)+example_number+".eps", bbox_inches='tight')
plt.show()