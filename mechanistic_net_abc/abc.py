# -*- coding: utf-8 -*-
"""
Implementation of various ABC algorithms, to determine the posteriors of the parameters.
"""

from mechanistic_net_abc.utility import simulate_weights, simulate_weights_truncated, _merge_dict, _is_discrete
from mechanistic_net_abc.models import mixture_model_simulation
from mechanistic_net_abc.summaries import compute_many_summaries_undirected
from mechanistic_net_abc.summaries import compute_indexed_summaries_undirected
from mechanistic_net_abc.utility import _perturb_continuous_param_on_support, _perturb_discrete_param_on_support

import numpy as np
import pandas as pd
import scipy.stats as ss

def distance_euclidean_std(df_sim_summaries, df_obs_summaries, std_values):
    dist = np.sqrt(np.sum(np.array( ( (df_sim_summaries.iloc[0,:] - df_obs_summaries.iloc[0,:])/std_values.iloc[0,:] )**2 )))
    return dist

def abc_RSMCABC(G_seed, num_nodes, func_mechanisms,
                prior_args_mechanisms=None, fixed_args_mechanisms=None,
                min_weight=None, max_weight=None, threshold_init=1,
                threshold_final=0, alpha=0.1, scale_factor=2,
                weight_perturbation="Gaussian",
                num_acc_sim=100, df_observed_summaries=None,
                distance_func=None, distance_args=None,
                sel_sum_names=None, max_sim=50000):
    
    """ Implementation of the replenishment SMC ABC algorithm.
    
    We here implement the replenishment SMC ABC algorithm, proposed by Drovandi
    and Pettitt, (2011).
    
    Drovandi, C. C. and Pettitt, A. N. "Estimation of Parameters for
    Macroparasite Population Evolution Using Approximate Bayesian Computation"
    Biometrics, 67, 225-233, (2011).
    
    Args:
        G_seed (networkx.classes.graph.Graph):
            a networkx graph to update according to the mixture of mechanisms.
        num_nodes (int):
            the number of nodes in the simulated network.
        func_mechanisms (list of functions):
            a list of functions, where each corresponds to the application of a
            mechanism.
        prior_args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G), and each
            values is a scipy.stats frozen distribution 
            (scipy.stats._distn_infrastructure.rv_frozen),
            in this case the .rvs() attribute of this object will be used to
            simulate a parameter value according to this prior distribution.
            For example, if for the first mechanism, m has as prior a uniform 
            discrete distribution on [1,2]; and the second mechanism a uniform
            discrete distribution on [1,3], it gives:
                dist1 = scipy.stats.randint(1,3)
                dist2 = scipy.stats.randint(1,4)
                prior_args_mechanisms = [{'m':dist1}, {'m':dist2}].
        fixed_args_mechanisms (list of dict):
            a list of dictionaries where each element of the list corresponds to
            a dictionary associated to a mechanism, where the keys of the dict
            are the arguments of the mechanism function (other than G) that
            do not have a prior function on them, and the values are the
            fixed parameter values.
        min_weight (list):
            each component indicates the minimal possible value for a 
            corresponding mechanism weight.
        max_weight (list):
            each component indicates the maximal possible value for a 
            corresponding mechanism weight.
        threshold_init (float):
            the initial threshold epsilon under which a simulated data is judged
            acceptable for the first step of this SMC ABC algorithm.
        threshold_final (float):
            the final threshold epsilon under which a simulated data is judged
            acceptable for the last step of this SMC ABC algorithm.
        alpha (float):
            a float between 0 and 1, used for discarding 100*alpha% of the
            particles at each step of the SMC algorithm.
        scale_factor (float):
            scaling factor of the covariance matrix of the Gaussian perturbation
            kernel, or of the Dirichlet parameter (see below).
        weight_perturbation (str):
            which method to use for the weight perturbation. If "Gaussian", then
            a multivariate Gaussian distribution is used, centered at the weight
            values (w_old,i) to perturb, with covariance matrix, scale_factor * the empirical
            covariance matrix computed on the particles accepted at the previous step
            of the algorithm. If "Dirichlet", then a multivariate Dirichlet 
            distribution is used for perturbation, where each parameters alpha_i
            is cov_fac * w_old,i + 1, so that the mode of the Dirichlet distribution
            is (w_old,i) and cov_fac tunes the spread of the distribution.
        num_acc_sim (int):
            the number of accepted parameters values at each step of the algorithm,
            before discarding 100*alpha% of the particles.
        df_observed_summaries (pandas.core.frame.DataFrame):
            a pandas data frame of the observed summary statistics.
        distance_func (function):
            a distance function to be computed between two pandas data frames.
        distance_args (dict):
            a dictionary that contains additional argument values for
            distance_func. Each key is an argument, and each value is the
            corresponding argument value.
        sel_sum_names (list or numpy.nparray):
            if None, all the possible summaries available in the function
            compute_many_summaries_undirected are computed (i.e. 46),
            else only the summaries with the specified names are computed.
            This is useful once selection has been performed and interesting
            summaries have been identified.
        max_sim (int):
            the maximum number of simulations, after that number the algorithm
            finishes its current step and return the results.
            
    Returns:
        df_weights (pandas.core.frame.DataFrame):
            a pandas dataframe that contains the accepted weight values for the
            mechanisms.
        df_params (pandas.core.frame.DataFrame):
            a pandas dataframe that contains the accepted parameter values for 
            the mechanisms.
        df_dist_acc (pandas.core.frame.DataFrame):
            a pandas dataframe that contains the corresponding distances
            computed.
        sim_count_total (int):
            the total number of data simulation performed in the algorithm.
        threshold_values (numpy.ndarray):
            the sequence of distance thresholds used in the algorithm.
            
    """
    
    num_mech = len(func_mechanisms)
    
    if min_weight == None:
        min_weight = [0]*num_mech
    if max_weight == None:
        max_weight = [1]*num_mech
    
    names_sum_cpt = sel_sum_names
    if isinstance(names_sum_cpt, np.ndarray):
        names_sum_cpt = list(sel_sum_names).copy()
    
    # Prior used for the weights
    prior_weights = ss.dirichlet(num_mech*[1])
    
    # Recover a list for the parameter priors
    list_priors = []
    mechs_priors_identifier = []
    disc_identifier = []
    for idx in range(num_mech):
        # For each key, we recover from the prior
        for (key, value) in prior_args_mechanisms[idx].items():
            list_priors += [value]
            mechs_priors_identifier += [idx]
            disc_identifier += [_is_discrete(value)]
    
    # Uniform distribution for MH test
    unif_dist = ss.uniform(loc = 0, scale = 1)
    
    # Number of particles to discard at each step
    num_drop_sim = int(alpha * num_acc_sim)
    if num_drop_sim == 0:
        num_drop_sim = 1
    
    # Identify the summary statistics to keep while simulating
    cols_to_keep = df_observed_summaries.columns
    
    step_count = 0      # number of sequential steps
    sim_count_total = 0 # total number of simulated data
    
    # To store accepted weights/parameters values, and distances
    df_weights = pd.DataFrame()
    df_params = pd.DataFrame()
    df_dist_acc = pd.DataFrame()
    #df_summaries_acc = pd.DataFrame()
    
    # Keep track of the epsilon values
    epsilon_values = [threshold_init]
    
    if step_count == 0:
        
        sim_count = 0 # number of accepted simulations during the current step
        
        ### Initial classic rejection sampling algorithm
        while sim_count < num_acc_sim:
            
            sim_count_total += 1
            
            ### Simulate the weights and parameters
            
            # Generate the weights of the mechanisms
            if (True in [min_weight[i] > 0 for i in range(num_mech)]) or (True in [max_weight[i] < 1 for i in range(num_mech)]):
                weights = simulate_weights_truncated(num_mech, min_weight, max_weight)
            else:
                weights = simulate_weights(num_mech)
    
            # Generate parameters from the priors for each mechanism
            sim_values_args_mechanisms = [dict() for idx in range(num_mech)]
            
            for idx in range(num_mech):
                # For each key, we simulate from the prior
                for (key, value) in prior_args_mechanisms[idx].items():
                    if isinstance(value, ss._distn_infrastructure.rv_frozen):
                        sim_values_args_mechanisms[idx][key] = value.rvs()
                    else:
                        raise ValueError('Invalid specified value for the parameter prior distribution. Each prior must be a scipy.stats._distn_infrastructure.rv_frozen object.')
            
            # Fuse the dictionaries of simulated and fixed parameters together
            args_mechanisms = [_merge_dict(sim_values_args_mechanisms[idx],
                                           fixed_args_mechanisms[idx]) for idx in range(num_mech)]

            ### Simulate from the model given the simulated weights and params
            G_sim = mixture_model_simulation(G_seed, num_nodes, weights,
                                             func_mechanisms, args_mechanisms)

            ### Compute the summary statistics of the simulated data
            if names_sum_cpt is None:
                dict_summaries = compute_many_summaries_undirected(G_sim)
            elif isinstance(names_sum_cpt, list):
                dict_summaries = compute_indexed_summaries_undirected(G_sim, names_sum_cpt)
            else:
                raise ValueError('Invalid specified value for sel_sum_names. It should be a list, numpy array or None.')

            ### Format the weights to be stored
            dict_weights = {'weight_mech_'+str(idx+1): weights[idx] for idx in range(num_mech)}
        
            ### Format the parameter values simulated
            dict_params = {}
            for idx in range(num_mech):
                for (key, value) in sim_values_args_mechanisms[idx].items():
                    dict_params[key+'_mech_'+str(idx+1)] = value

            ### Convert the simulated network summaries and reduce it if necessary
            df_summaries = pd.DataFrame([dict_summaries])
            df_summaries_reduced = df_summaries[cols_to_keep]

            ### Compute the distance between simulated and observed data summarized
            dist = distance_func(df_summaries_reduced, df_observed_summaries, **distance_args)
            
            # If the distance is low enough, accept the simulated parameters
            if dist <= threshold_init:
    
                #df_weights = pd.DataFrame([dict_weights])
                #df_params = pd.DataFrame([dict_params])
                #df_w_p = pd.concat([df_weights, df_params],
                #                   axis=1, join="inner")
                #df_weights_params_acc = df_weights_params_acc.append(df_w_p)
                
                # store also the accepted weights/parameters and resulting distance
                df_weights = df_weights.append(pd.DataFrame([dict_weights]), ignore_index=True)
                df_params = df_params.append(pd.DataFrame([dict_params]), ignore_index=True)
                df_dist_acc = df_dist_acc.append(pd.DataFrame([dist]), ignore_index=True)
                
                sim_count += 1
        
        step_count += 1
    
    # SMC-ABC core part
    if step_count > 0:
        
        # Determine the order of the distances when sorted in increasing order
        idx_sort = np.argsort(df_dist_acc.iloc[:,0])
        
        # Reorder the parameters and distance with this order
        df_dist_acc = df_dist_acc.iloc[idx_sort,:]
        df_dist_acc = df_dist_acc.reset_index(drop=True)
        
        df_weights = df_weights.iloc[idx_sort,:]
        df_weights = df_weights.reset_index(drop=True)
        
        df_params = df_params.iloc[idx_sort,:]
        df_params = df_params.reset_index(drop=True)
        
        # Compute epsilon_max = the maximal distance
        epsilon_max = df_dist_acc.iloc[num_acc_sim-1,0]
        
        epsilon_values = epsilon_values + [epsilon_max]
        
        # while epsilon_max is greater than epsilon_final
        while (epsilon_max > threshold_final) and (sim_count_total < max_sim):
            
            print(epsilon_max, threshold_final)
            
            # Drop the num_drop_sim (Na) particles with largest distances
            df_weights.drop(df_dist_acc.tail(num_drop_sim).index, inplace=True)
            df_params.drop(df_dist_acc.tail(num_drop_sim).index, inplace=True)
            df_dist_acc.drop(df_dist_acc.tail(num_drop_sim).index, inplace=True)
            
            epsilon_next = df_dist_acc.tail(1).iloc[0,0] # the largest distance of the remaining simulations
            
            if weight_perturbation == "Gaussian":
                # Compute the covariance matrix of the perturbation kernel
                cov_mat = scale_factor * np.cov(df_weights, rowvar=False)
            
            # !!! TODO: use a different scale_factor when Dirichlet is used
            std_params = np.sqrt(scale_factor) * df_params.apply(np.std)
            
            ### Resample num_drop_sim new particles and data that are accepted
            
            num_acc_next = 0
            
            while num_acc_next < num_drop_sim:
            
            #for j in range(num_drop_sim):
                
                ### Sample an old weight and parameter value from the
                ### num_acc_sim - num_drop_sim previously accepted values 
                idx_sel = np.random.choice(df_weights.index[:(num_acc_sim-num_drop_sim)])
                sim_count_total += 1
                
                ### Perturb the selected weights and parameter values with a kernel
                
                prev_weight = np.array(df_weights.iloc[idx_sel,:])
                if weight_perturbation == "Gaussian":
                    # Strategy 2:
                    # To perturb the weights, let's use a multidimensional Gaussian 
                    # kernel centered in the previous weight values and with matrix
                    # of variance covariance, scale_factor times the empirical matrix
                    # of variance-covariance of the previous weights
                    perturbation_kernel = ss.multivariate_normal(mean = prev_weight,
                                                                 cov = cov_mat,
                                                                 allow_singular = True)
                    perturbed_weight = perturbation_kernel.rvs()
                    # The perturbed weight, and the previous weights do not necessarily
                    # sum to 1, (numerical issues I guess), we devide by the sum then
                    perturbed_weight = perturbed_weight / perturbed_weight.sum()

                    # By adding this condition, we consider a truncated Gaussian distribution
                    # between min_weight and max_weight, usually 0 and 1
                    while (True in [perturbed_weight[i]<min_weight[i] for i in range(num_mech)]) or (True in [perturbed_weight[i]>max_weight[i] for i in range(num_mech)]):
                        perturbed_weight = perturbation_kernel.rvs()
                        perturbed_weight = perturbed_weight / perturbed_weight.sum()

                elif weight_perturbation == "Dirichlet":
                    # Strategy 3:
                    # Use a Dirichlet with mode the previous weights, by using
                    # as parameters beta * weights_prev + 1
                    params_dirichlet_prev = scale_factor * prev_weight + 1
                    perturbation_kernel = ss.dirichlet(alpha = params_dirichlet_prev)
                    perturbed_weight = perturbation_kernel.rvs()[0]
                    perturbed_weight = perturbed_weight / perturbed_weight.sum()

                else:
                    raise ValueError('Invalid value entered for weight_perturbation. It must be either "Gaussian" or "Dirichlet".')
                
                # Parameter perturbation
                prev_params = np.array(df_params.iloc[idx_sel,:])
                
                perturbed_params = np.empty(len(prev_params))
                # For each parameter value
                for i in range(len(prev_params)):
                    perturbation_kernel_Gauss = ss.norm(prev_params[i], std_params[i])
                    # if the parameter is discrete, we use a discretized Gaussian on the support of the prior
                    if disc_identifier[i]:
                        perturbed_params[i] = _perturb_discrete_param_on_support(list_priors[i], perturbation_kernel_Gauss)
                    # else we use a continuous Gaussian on the support of the prior
                    else:
                        perturbed_params[i] = _perturb_continuous_param_on_support(list_priors[i], perturbation_kernel_Gauss)
                    
                #perturbed_params = np.array(df_params.iloc[idx_sel,:])
                
                # To use the simulated parameters in our data generation function
                # we need a list of dict, with same structure as sim_args_mechanisms
                perturbed_params_list_dict = [dict() for i in range(num_mech)]
                idx_params = 0
                for idx in range(num_mech):
                    for (key, value) in sim_values_args_mechanisms[idx].items():
                        # if the parameter is discrete, we need an integer for the mechanisms
                        if disc_identifier[idx_params]:
                            perturbed_params_list_dict[idx][key] = int(perturbed_params[idx_params])
                        else:
                            perturbed_params_list_dict[idx][key] = perturbed_params[idx_params]
                        idx_params += 1
                
                ### Generate a new data given the perturbed parameters
                args_mechanisms = [_merge_dict(perturbed_params_list_dict[idx],
                                               fixed_args_mechanisms[idx]) for idx in range(num_mech)]

                G_sim = mixture_model_simulation(G_seed, num_nodes, perturbed_weight,
                                                 func_mechanisms, args_mechanisms)
                
                if names_sum_cpt is None:
                    dict_summaries = compute_many_summaries_undirected(G_sim)
                elif isinstance(names_sum_cpt, list):
                    dict_summaries = compute_indexed_summaries_undirected(G_sim, names_sum_cpt)
                else:
                    raise ValueError('Invalid specified value for sel_sum_names. It should be a list, numpy array or None.')
           
                df_summaries = pd.DataFrame([dict_summaries])
                df_summaries_reduced = df_summaries[cols_to_keep]
    
                dist_new = distance_func(df_summaries_reduced,
                                         df_observed_summaries,
                                         **distance_args)
                
                if dist_new <= epsilon_next:
                    
                    print("Dist_new: ", dist_new, " Epsilon next: ", epsilon_next)
                                        
                    # !!! When sampling parameters too, include the ratio
                    # prior(new_params) / prior(old_param_sampled) for the mechanism parameters
                    # and the transition part for them too.
                    if weight_perturbation == "Gaussian":
                        pdf_value_prev_given_new_weights = ss.multivariate_normal.pdf(prev_weight,
                                                                              mean = perturbed_weight,
                                                                              cov = cov_mat,
                                                                              allow_singular = True)
                    elif weight_perturbation == "Dirichlet":
                        params_dirichlet_new = scale_factor * perturbed_weight + 1
                        pdf_value_prev_given_new_weights = ss.dirichlet.pdf(prev_weight,
                                                                    alpha = params_dirichlet_new)                        

                    # The simulation/perturbation of the weights ensures that the pdf
                    # is already evaluated in the truncated Dirichlet,
                    # since we will never evaluate on the zero density parts
                    prior_ratio_weights = prior_weights.pdf(perturbed_weight) / prior_weights.pdf(prev_weight)
                    transition_ratio_weights = pdf_value_prev_given_new_weights / perturbation_kernel.pdf(perturbed_weight)
                    
                    # For the parameters
                    list_prior_params_old = []
                    list_prior_params_new = []
                    list_pdf_new_given_old = []
                    list_pdf_old_given_new = []
                    for i in range(len(list_priors)):
                        if disc_identifier[i]:
                            list_prior_params_old += [list_priors[i].pmf(prev_params[i])]
                            list_prior_params_new += [list_priors[i].pmf(perturbed_params[i])]   
                            list_pdf_new_given_old += [1 if std_params[i] == 0
                                                       else ss.norm(prev_params[i], std_params[i]).cdf(perturbed_params[i]+0.5) - ss.norm(prev_params[i], std_params[i]).cdf(perturbed_params[i]-0.5)]
                            list_pdf_old_given_new += [1 if std_params[i] == 0 
                                                       else ss.norm(perturbed_params[i], std_params[i]).cdf(prev_params[i]+0.5) - ss.norm(perturbed_params[i], std_params[i]).cdf(prev_params[i]-0.5)]
                        else:
                            list_prior_params_old += [list_priors[i].pdf(prev_params[i])]
                            list_prior_params_new += [list_priors[i].pdf(perturbed_params[i])]
                            list_pdf_new_given_old += [ss.norm(prev_params[i], std_params[i]).pdf(perturbed_params[i])]
                            list_pdf_old_given_new += [ss.norm(perturbed_params[i], std_params[i]).pdf(prev_params[i])]

                    prior_ratio_params = np.prod(list_prior_params_new) / np.prod(list_prior_params_old)
                    transition_ratio_params = np.prod(list_pdf_old_given_new) / np.prod(list_pdf_new_given_old)
                    
                    mh_ratio = np.min([1, prior_ratio_weights * prior_ratio_params * transition_ratio_weights * transition_ratio_params])
                    
                    if unif_dist.rvs() < mh_ratio:

                        perturbed_weight_df = pd.DataFrame(perturbed_weight.reshape(-1, len(perturbed_weight)),columns=df_weights.columns)
                        df_weights = df_weights.append(perturbed_weight_df, ignore_index=True)
                        if len(perturbed_params) > 0:
                            perturbed_params_df = pd.DataFrame(perturbed_params.reshape(-1, len(perturbed_params)),columns=df_params.columns)
                            df_params = df_params.append(perturbed_params_df, ignore_index=True)
                        else:
                            df_params = df_params.append(pd.DataFrame([], index=[1]), ignore_index=True)
                        df_dist_acc = df_dist_acc.append(pd.DataFrame([dist_new]), ignore_index=True)
                        
                        num_acc_next += 1
                        
            # Determine the order of the distances when sorted in increasing order
            idx_sort = np.argsort(df_dist_acc.iloc[:,0])
            
            # Reorder the parameters and distance with this order
            df_dist_acc = df_dist_acc.iloc[idx_sort,:]
            df_dist_acc = df_dist_acc.reset_index(drop=True)
            
            df_weights = df_weights.iloc[idx_sort,:]
            df_weights = df_weights.reset_index(drop=True)
            
            df_params = df_params.iloc[idx_sort,:]
            df_params = df_params.reset_index(drop=True)
            
            # Compute epsilon_max = the maximal distance
            epsilon_max = df_dist_acc.iloc[num_acc_sim-1,0]
            
            epsilon_values = epsilon_values + [epsilon_max]
            
            step_count += 1
            
        threshold_values = np.array(epsilon_values)

        return df_weights, df_params, df_dist_acc, sim_count_total, threshold_values
    