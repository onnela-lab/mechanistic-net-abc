# -*- coding: utf-8 -*-
"""
Implementation of the recursive summary statistic selection technique based on random forest.
Step 1 :
    Fit a random forest (muti-output if necessary) with default parameters,
    and compute the Mean Decrease of Accuracy (MDA) to obtain a ranking of the
    summary statistics.
Step 2 :
    Use Principal Component Analysis (PCA) axes to reduce the summary space 
    (number of components that preserve at least 90% of the total variance).
Step 3 :
    Find n_s simulated data similar to the observed data on this reduced data space.
Step 4 :
    For different numbers of selected summary statistics (from 1 to num_sums)
    fit a K-NN-ABC algorithm on each of these n_s data for which we know the true
    parameters.
    For each data l and each number of summaries, compute the corresponding RMSE:
        sqrt( 1/K * \sum_{k=1}^K ( theta_k - theta_{true,l} )^2 ).
    Average these RMSE over these n_s data, to obtain the local error
    associated to this number of selected summaries.
    Choose the correct number of selected summaries based on the errors.
"""

from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

##### Step 1: Random forest training and MDA deduction #####

def random_forest_MDA_reg(rf_model,
                          covariates_train,
                          responses_scaled_train,
                          covariates_val,
                          responses_scaled_val,
                          n_repeats=10, random_state=123):
    """ 
    Train an RF and deduce the summary statistic ranking from permutation importance (a.k.a
    mean decrease of accuracy, MDA)
        
    Args:
        rf_model (sklearn.ensemble._forest.RandomForestRegressor):
            a sklearn random forest regressor with default parameters.
        covariates_train (numpy.ndarray):
            summary statistics (features) to train the random forest.
        response_scaled_train (numpy.ndarray):
            scaled parameter values (response variables) to train the random forest.
        covariates_val (numpy.ndarray):
            summary statistic values for permutation importance calculation.
        response_scaled_val (numpy.ndarray):
            scaled parameter values for permutation importance calculation.
        n_repeats (int):
            number of repeats for permutation importance calculation.
        random_state (int):
            random seed for permutation importance calculation.
                    
    Returns:
        rf_MDA_rankings (list):
            the random forest ranking deduced from permutation importance.        
                
    """

    # Train the random forest
    rf_model.fit(covariates_train, responses_scaled_train)

    # Compute the MDA
    res_permutation_importance = \
    permutation_importance(rf_model, covariates_val, responses_scaled_val,
                           scoring='neg_mean_squared_error',
                           n_repeats=n_repeats, random_state=random_state)
    
    # Deduce the ranking
    rf_MDA_rankings = res_permutation_importance.importances_mean.argsort()[::-1]

    return(rf_MDA_rankings)


##### Step 2 : Use PCA to reduce the summary statistic space #####
    
def fit_transform_PCA(covariates_train, covariates_obs, pvar_min = 0.90):
    
    num_summaries = covariates_train.shape[1]
    pca_model = PCA(n_components = num_summaries)
    
    pca_model.fit(covariates_train)
    
    num_compo = np.arange(1,num_summaries+1)[np.cumsum(pca_model.explained_variance_ratio_)>=pvar_min][0]
    
    covariates_train_PCA = pca_model.transform(covariates_train)[:,range(num_compo)]
    covariates_obs_PCA = pca_model.transform(covariates_obs)[:,range(num_compo)]
    
    return(covariates_train_PCA, covariates_obs_PCA)

##### Step 3 : Find the closest data from the observed one in PCA dim #####

def identify_neighbors_PCA(covariates_train_PCA, covariates_obs_PCA, num_neigh=100):

    nearest_neigh = NearestNeighbors(n_neighbors=num_neigh, algorithm="brute")
    nearest_neigh.fit(covariates_train_PCA)
    distances_pseudo_obs, indices_pseudo_obs = nearest_neigh.kneighbors(covariates_obs_PCA)

    return(indices_pseudo_obs[0])

##### Step 4 : #####

# For a given neighbor of the observed data, fit the K-NN-ABC algorithm and compute the RMSE
# Covariates should be scaled, and RMSE should be standardized by the prior standard deviation.

def select_summaries(covariates_knn, responses_scaled_knn,
                     covariates_train, responses_scaled_train,
                     ranking, indices_pseudo_obs, n_neighbors=200, pred_type='individual',
                     plus_one_std_error=False):
    """
    Set pred_type = 'average' or pred_type='individual'
    If 'average': 
        the average value of the neighboring data is compared to the pseudo
        observed parameter value.
    If 'individual':
        all the neighboring parameter values are compared to the pseudo
        observed parameter value.
    """
    
    num_covariates = covariates_knn.shape[1]
    num_responses = responses_scaled_knn.shape[1]
    average_RMSE_total_condSel = []
    average_RMSE_perResponse_condSel = []
    size_sel_possible = np.arange(1,num_covariates+1)
    
    for K in size_sel_possible:
        
        covariates_subset = ranking[:K]
        RMSE_total = []
        RMSE_perResponse = []
    
        nearest_neigh_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
        nearest_neigh_model.fit(covariates_knn.iloc[:,covariates_subset])
    
        for idx_obs in indices_pseudo_obs:
            
            # Search for the k-NN indices from covariates_knn data
            distances_obs, indices_obs = nearest_neigh_model.kneighbors(covariates_train.iloc[:,covariates_subset].iloc[[idx_obs]])
            
            # Use the df_weights_params together to identify the accpeted parameters (std)
            if pred_type=='individual':
                accepted_responses_std = responses_scaled_knn.iloc[indices_obs[0],:]
            elif pred_type=='average':
                accepted_responses_std = responses_scaled_knn.iloc[indices_obs[0],:].apply(np.mean).to_frame().transpose()
            
            true_responses_std = responses_scaled_train.iloc[[idx_obs]]
            
            # Compute the RMSE = average RMSE (aRMSE) over all parameters
            true_to_compare = pd.DataFrame()
            true_to_compare = true_to_compare.append([true_responses_std]*accepted_responses_std.shape[0], ignore_index=True)
            RMSE_total += [mean_squared_error(true_to_compare, accepted_responses_std, squared=False)]
            for i in range(num_responses):
                RMSE_perResponse += [mean_squared_error(true_to_compare, accepted_responses_std, multioutput='raw_values', squared=False).tolist()]
            
        # Compute the average aRMSE over all the similar data to the observed one
        average_RMSE_total_condSel += [np.mean(RMSE_total)]
        average_RMSE_perResponse_condSel += [np.array(RMSE_perResponse).mean(axis=0).tolist()]
    
    # Find the best number of selected summary statistics
    if plus_one_std_error is False:
        num_sel_cov_final = size_sel_possible[list(average_RMSE_total_condSel).index(np.min(average_RMSE_total_condSel))]
    else:
        std_error = np.std(average_RMSE_total_condSel)
        thresh_acceptable_error = np.min(average_RMSE_total_condSel)+std_error
        mask_acceptable_errors = np.array(average_RMSE_total_condSel)<=thresh_acceptable_error
        num_sel_cov_final = size_sel_possible[mask_acceptable_errors][0] # Keep the first size
    
    return(average_RMSE_perResponse_condSel, average_RMSE_total_condSel, num_sel_cov_final)


### Final function to combine everything ###

def RFMDA_select_summaries(rf_model, covariates_train, responses_scaled_train,
                           covariates_val, responses_scaled_val,
                           covariates_knn, responses_scaled_knn,
                           covariates_obs,
                           n_repeats_MDA=10, random_state_MDA=123,
                           pvar_min_PCA=0.90, num_neigh_PCA=100,
                           num_neighbors_KNN=500, pred_type='individual',
                           plus_one_std_error=False):

    # Find neighbors of the observed data to evaluated ABC performance on them    
    covariates_train_PCA, covariates_obs_PCA = \
        fit_transform_PCA(covariates_train=covariates_train,
                          covariates_obs=covariates_obs,
                          pvar_min=pvar_min_PCA)
    
    indices_pseudo_obs = \
        identify_neighbors_PCA(covariates_train_PCA=covariates_train_PCA,
                               covariates_obs_PCA=covariates_obs_PCA,
                               num_neigh=num_neigh_PCA)

    ranking_RFMDA = random_forest_MDA_reg(rf_model=rf_model,
                                          covariates_train=covariates_train,
                                          responses_scaled_train=responses_scaled_train,
                                          covariates_val=covariates_val,
                                          responses_scaled_val=responses_scaled_val,
                                          n_repeats=n_repeats_MDA, 
                                          random_state=random_state_MDA)
    
    average_RMSE_perResponse_condSel, average_RMSE_total_condSel, num_sel_sums_final =\
        select_summaries(covariates_knn=covariates_knn,
                         responses_scaled_knn=responses_scaled_knn,
                         covariates_train=covariates_train,
                         responses_scaled_train=responses_scaled_train,
                         ranking=ranking_RFMDA, 
                         indices_pseudo_obs=indices_pseudo_obs,
                         n_neighbors=num_neighbors_KNN,
                         pred_type=pred_type,
                         plus_one_std_error=plus_one_std_error)

    selected_covariates = ranking_RFMDA[:num_sel_sums_final]
    return(average_RMSE_perResponse_condSel, average_RMSE_total_condSel, num_sel_sums_final, selected_covariates)


########### Implementation of recurive feature elimination ###########

def recursiveElimination_RFMDA_select_summaries(covariates_train,
                                                responses_scaled_train,
                                                covariates_val, responses_scaled_val,
                                                covariates_knn, responses_scaled_knn,
                                                covariates_obs,
                                                n_estimators=500,
                                                max_features='auto',
                                                n_repeats_MDA=10,
                                                random_state_MDA=123,
                                                pvar_min_PCA=0.90,
                                                num_neigh_PCA=100,
                                                num_neighbors_KNN=200,
                                                pred_type='individual',
                                                n_jobs=1,
                                                plus_one_std_error=False):
    """
    Implementation of the recursive selection algorithm based on multi-output
    random forest and local ABC errors.

    Args:
        covariates_train (numpy.ndarray):
            summary statistics (features) to train the random forest.
        response_scaled_train (numpy.ndarray):
            scaled parameter values (response variables) to train the random forest.
        covariates_val (numpy.ndarray):
            summary statistic values for permutation importance calculation.
        response_scaled_val (numpy.ndarray):
            scaled parameter values for permutation importance calculation.
        covariates_knn (numpy.ndarray):
            summary statistic values for K-NN-ABC.
        response_scaled_knn (numpy.ndarray):
            scaled parameter values for local error computation.
        num_estimators (int):
            number of tree in the random forest.
        max_features (int or str):
            the number features sampled at each tree node for random forest training.
        n_repeats_MDA (int):
            number of repeats for permutation importance calculation.
        random_state_MDA (int):
            random seed for permutation importance calculation.
        pvar_min_PCA (float):
            the minimal percentage of explained variance to preserve from PCA.
        num_neigh_PCA (int):
            number of simulated data similar to the observation on which the
            local errors are computed.
        pred_type (str):
            'individual' or 'average'. 'individual' compares the every K-NN-ABC output
            with one of the num_neigh_PCA data on which this ABC algorithm was fitted.
            'average' computes the average of the K-NN-ABC output before comparison.
        n_jobs (int):
            number of CPU cores to use for random forest training.
        plus_one_std_error (bool):
            should the plus one standard error rule be applied to select
            the best subset of summary statistics.

    Returns:
        average_RMSE_perResponse_complete (list):
            values of the parameter specific average local error. This is a list of list, where for 
            a given list, the i-th indice corresponds to i-1 discarded summaries.
        average_RMSE_total_complete (list):
            values of the average local error.
            At the i-th index, i-1 summaries have been discarded.
        eliminated_features (list):
            list of the eliminated summary statistics.
        recursive_selected_summaries (list):
            list of the selected summaries.

    """

    # Find neighbors of the observed data to evaluated ABC performance on them    
    covariates_train_PCA, covariates_obs_PCA = \
        fit_transform_PCA(covariates_train=covariates_train,
                          covariates_obs=covariates_obs,
                          pvar_min=pvar_min_PCA)
    
    indices_pseudo_obs = \
        identify_neighbors_PCA(covariates_train_PCA=covariates_train_PCA,
                               covariates_obs_PCA=covariates_obs_PCA,
                               num_neigh=num_neigh_PCA)

    num_covariates = covariates_train.shape[1]
    set_cov_toKeep = np.arange(0,covariates_train.shape[1])
    eliminated_features = []

    average_RMSE_total_complete = []
    average_RMSE_perResponse_complete = []

    for i in range(num_covariates):
        
        if i == 0:
            
            # Preliminary step when not discarding any covariates
            average_RMSE_total_thisSubset, average_RMSE_perResponse_thisSubset =\
                evaluate_perf(covariates_knn=covariates_knn.iloc[:,set_cov_toKeep],
                              responses_scaled_knn=responses_scaled_knn,
                              covariates_train=covariates_train.iloc[:,set_cov_toKeep],
                              responses_scaled_train=responses_scaled_train,
                              indices_pseudo_obs=indices_pseudo_obs,
                              n_neighbors=num_neighbors_KNN,
                              pred_type=pred_type)
            average_RMSE_total_complete += average_RMSE_total_thisSubset
            average_RMSE_perResponse_complete += average_RMSE_perResponse_thisSubset            
            
        if i > 0:# and i!=(num_covariates)-1):
          
            print(i)
            rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                             criterion='mse',
                                             max_features=max_features,
                                             bootstrap=True,
                                             oob_score=False,
                                             n_jobs=n_jobs,
                                             random_state=123)
    
            ranking_RFMDA = random_forest_MDA_reg(rf_model=rf_model,
                                                  covariates_train=covariates_train.iloc[:,set_cov_toKeep],
                                                  responses_scaled_train=responses_scaled_train,
                                                  covariates_val=covariates_val.iloc[:,set_cov_toKeep],
                                                  responses_scaled_val=responses_scaled_val,
                                                  n_repeats=n_repeats_MDA,
                                                  random_state=random_state_MDA)
                        
            ranking_starting_column_index = set_cov_toKeep[ranking_RFMDA]
            # Eliminate the feature from the list
            print("before: ", set_cov_toKeep)
            set_cov_toKeep = np.delete(set_cov_toKeep,
                                       np.where(set_cov_toKeep == ranking_starting_column_index[-1]))
            print("after: ", set_cov_toKeep)
            # Add the eliminated feature in the corresponding list
            eliminated_features += [ranking_starting_column_index[-1]]
            print("eliminated_features: ", eliminated_features)
            
            # Compute the error on this subset        
            average_RMSE_total_thisSubset, average_RMSE_perResponse_thisSubset =\
                evaluate_perf(covariates_knn=covariates_knn.iloc[:,set_cov_toKeep],
                              responses_scaled_knn=responses_scaled_knn,
                              covariates_train=covariates_train.iloc[:,set_cov_toKeep],
                              responses_scaled_train=responses_scaled_train,
                              indices_pseudo_obs=indices_pseudo_obs,
                              n_neighbors=num_neighbors_KNN,
                              pred_type=pred_type)
    
            average_RMSE_total_complete += average_RMSE_total_thisSubset
            average_RMSE_perResponse_complete += average_RMSE_perResponse_thisSubset

    if plus_one_std_error is False:
        idx_min_error = average_RMSE_total_complete.index(np.min(average_RMSE_total_complete))
        to_eliminate_summaries = eliminated_features[:idx_min_error]
        recursive_selected_summaries = np.delete(np.arange(num_covariates),
                                                 to_eliminate_summaries)
    else:
        std_error = np.std(average_RMSE_total_complete)
        thresh_acceptable_error = np.min(average_RMSE_total_complete)+std_error
        mask_acceptable_errors = np.array(average_RMSE_total_complete)<=thresh_acceptable_error
        idx_min_error = np.arange(num_covariates)[mask_acceptable_errors][-1]
        to_eliminate_summaries = eliminated_features[:idx_min_error]
        recursive_selected_summaries = np.delete(np.arange(num_covariates),
                                                 to_eliminate_summaries)

    return(average_RMSE_perResponse_complete, average_RMSE_total_complete, eliminated_features, recursive_selected_summaries)



def evaluate_perf(covariates_knn, responses_scaled_knn,
                  covariates_train, responses_scaled_train,
                  indices_pseudo_obs, n_neighbors=200,
                  pred_type='individual'):
    """
    Set pred_type = 'average' or pred_type='individual'
    If 'average': 
        the average value of the neighboring data is compared to the pseudo
        observed parameter value.
    If 'individual':
        all the neighboring parameter values are compared to the pseudo
        observed parameter value. 
    """
    
    num_responses = responses_scaled_knn.shape[1]
    RMSE_total = []
    RMSE_perResponse = []
    
    nearest_neigh_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    nearest_neigh_model.fit(covariates_knn)
    
    # For each data similar to the observed one, we use an ABC algo (knn here)
    for idx_obs in indices_pseudo_obs:
            
        # Search for the k-NN indices from covariates_knn data
        distances_obs, indices_obs =\
            nearest_neigh_model.kneighbors(covariates_train.iloc[[idx_obs]])
            
        # Use the df_weights_params together to identify the accpeted parameters (std)
        # accepted_weights_params_std = responses_knn.iloc[indices_obs[0],:]/std_prior_params
        if pred_type=='individual':
            accepted_responses_std = responses_scaled_knn.iloc[indices_obs[0],:]
        elif pred_type=='average':
            accepted_responses_std = responses_scaled_knn.iloc[indices_obs[0],:].apply(np.mean).to_frame().transpose()
            
        # Standardize the true response to give equal weights in RMSE computation
        # true_weights_params_std = responses_train.iloc[idx_obs,:]/std_prior_params
        true_responses_std = responses_scaled_train.iloc[[idx_obs]]
            
        # Compute the RMSE = average RMSE (aRMSE) over all parameters
        true_to_compare = pd.DataFrame()
        true_to_compare = true_to_compare.append([true_responses_std]*accepted_responses_std.shape[0], ignore_index=True)
        RMSE_total += [mean_squared_error(true_to_compare, accepted_responses_std, squared=False)]
        for i in range(num_responses):
            RMSE_perResponse += [mean_squared_error(true_to_compare, accepted_responses_std, multioutput='raw_values', squared=False).tolist()]
            
    average_RMSE_total_thisSubset = [np.mean(RMSE_total)]
    average_RMSE_perResponse_thisSubset = [np.array(RMSE_perResponse).mean(axis=0).tolist()]
    return(average_RMSE_total_thisSubset, average_RMSE_perResponse_thisSubset)
    