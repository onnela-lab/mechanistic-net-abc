# -*- coding: utf-8 -*-
"""
Function to compute summary statistics of a networkx graph.
"""

import networkx as nx
import numpy as np
import scipy.stats as ss
from networkx.algorithms import approximation

def compute_summaries_undirected(G):
    """ Compute network features (a small number).
    
    This function evaluates 9 summary statistics of an undirected networkx
    graph G. 
    (Number of edges, number of connected components, number of nodes in the 
     largest connected component (LCC), diameter of the LCC, average degree of 
     the neighborhood of each node, mean, max and standard deviation of the 
     degree distribution, number of triangles.)
        
        Args:
            G (networkx.classes.graph.Graph):
                an undirected networkx graph.
        
        Returns:
            dictSums (dict):
                a dictionary with the name of the summaries as keys and the
                summary statistic values as values.
                
    """
    
    dictSums = dict()   # To store the summary statistic values
                        
    # Extract the largest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G_lcc = G.subgraph(Gcc[0])

    # Number of edges
    dictSums["num_edges"] = G.number_of_edges()
    
    # Number of connected components
    dictSums["num_of_CC"] = nx.number_connected_components(G)
    
    # Number of nodes in the largest connected component
    dictSums["num_nodes_LCC"] = nx.number_of_nodes(G_lcc)
    
    # Diameter of the largest connected component
    dictSums["diameter_LCC"] = nx.diameter(G_lcc)
        
    # Average degree of the neighborhood of each node
    dictSums["avg_deg_connectivity"] = np.mean(list(nx.average_degree_connectivity(G).values()))
    
    # Recover the degree distribution
    degree_vals = list(dict(G.degree()).values())
    
    # Average degree
    dictSums["degree_mean"] = np.mean(degree_vals)

    # Max degree
    dictSums["degree_max"] = np.max(degree_vals)

    # Standard deviation of the degree distribution
    dictSums["degree_std"] = np.std(degree_vals)
    
    # Number of triangles
    dictSums["num_triangles"] = np.sum( list( nx.triangles(G).values() ) )/3
        
    return dictSums


def compute_many_summaries_undirected(G):
    """ Compute network features (a large number).
    
    This function evaluates 46 summary statistics of an undirected networkx 
    graph G. See our associated paper or code below for the list.
        
        Args:
            G (networkx.classes.graph.Graph):
                an undirected networkx graph.
        
        Returns:
            dictSums (dict):
                a dictionary with the name of the summaries as keys and the
                summary statistic values as values.
                
    """
    
    dictSums = dict()   # To store the summary statistic values
                        
    # Extract the largest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G_lcc = G.subgraph(Gcc[0])

    # Number of edges
    dictSums["num_edges"] = G.number_of_edges()
    
    # Number of connected components
    dictSums["num_of_CC"] = nx.number_connected_components(G)
    
    # Number of nodes in the largest connected component
    dictSums["num_nodes_LCC"] = nx.number_of_nodes(G_lcc)

    # Number of edges in the largest connected component
    dictSums["num_edges_LCC"] = G_lcc.number_of_edges()
    
    # Diameter of the largest connected component
    dictSums["diameter_LCC"] = nx.diameter(G_lcc)
    
    # Average geodesic distance (shortest path length in the LCC)
    dictSums["avg_geodesic_dist_LCC"] = nx.average_shortest_path_length(G_lcc)
    
    # Average degree of the neighborhood of each node
    dictSums["avg_deg_connectivity"] = np.mean(list(nx.average_degree_connectivity(G).values()))
    
    # Average degree of the neighbors of each node in the LCC
    dictSums["avg_deg_connectivity_LCC"] = np.mean(list(nx.average_degree_connectivity(G_lcc).values()))

    # Recover the degree distribution
    degree_vals = list(dict(G.degree()).values())
    
    # Entropy of the degree distribution
    dictSums["degree_entropy"] = ss.entropy(degree_vals)
    
    # Maximum degree
    dictSums["degree_max"] = max(degree_vals)

    # Average degree
    dictSums["degree_mean"] = np.mean(degree_vals)

    # Median degree
    dictSums["degree_median"] = np.median(degree_vals)

    # Standard deviation of the degree distribution
    dictSums["degree_std"] = np.std(degree_vals)

    # Quantile 25%
    dictSums["degree_q025"] = np.quantile(degree_vals, 0.25)
      
    # Quantile 75%
    dictSums["degree_q075"] = np.quantile(degree_vals, 0.75)
        
    # Average global efficiency:
    # The efficiency of a pair of nodes in a graph is the multiplicative 
    # inverse of the shortest path distance between the nodes.
    # The average global efficiency of a graph is the average efficiency of 
    # all pairs of nodes.
    dictSums["avg_global_efficiency"] = nx.global_efficiency(G)

    # Average local efficiency
    # The local efficiency of a node in the graph is the average global 
    # efficiency of the subgraph induced by the neighbors of the node. 
    # The average local efficiency is the average of the 
    # local efficiencies of each node.
    dictSums["avg_local_efficiency_LCC"] = nx.local_efficiency(G_lcc)
    
    # Node connectivity
    # The node connectivity is equal to the minimum number of nodes that 
    # must be removed to disconnect G or render it trivial.
    # Only on the largest connected component here.
    dictSums["node_connectivity_LCC"] = nx.node_connectivity(G_lcc)

    # Edge connectivity
    # The edge connectivity is equal to the minimum number of edges that 
    # must be removed to disconnect G or render it trivial.
    # Only on the largest connected component here.
    dictSums["edge_connectivity_LCC"] = nx.edge_connectivity(G_lcc)
    
    # Graph transitivity
    # 3*times the number of triangles divided by the number of triades
    dictSums["transitivity"] = nx.transitivity(G)
    
    # Number of triangles
    dictSums["num_triangles"] = np.sum( list( nx.triangles(G).values() ) )/3
    
    # Estimate of the average clustering coefficient of G:
    # Average local clustering coefficient, with local clustering coefficient
    # defined as C_i = (nbr of pairs of neighbors of i that are connected)/(nbr of pairs of neighbors of i)
    dictSums["avg_clustering_coef"] = nx.average_clustering(G)
      
    # Square clustering (averaged over nodes): 
    # the fraction of possible squares that exist at the node.
    
    # We average it over nodes
    dictSums["square_clustering_mean"] = np.mean( list( nx.square_clustering(G).values() ) )
 
    # We compute the median
    dictSums["square_clustering_median"] = np.median( list( nx.square_clustering(G).values() ) )

    # We compute the standard deviation
    dictSums["square_clustering_std"] = np.std( list( nx.square_clustering(G).values() ) )
    
    # Number of 2-cores
    dictSums["num_2cores"] = len(nx.k_core(G, k=2))
    
    # Number of 3-cores
    dictSums["num_3cores"] = len(nx.k_core(G, k=3))
    
    # Number of 4-cores
    dictSums["num_4cores"] = len(nx.k_core(G, k=4))
    
    # Number of 5-cores
    dictSums["num_5cores"] = len(nx.k_core(G, k=5))

    # Number of 6-cores
    dictSums["num_6cores"] = len(nx.k_core(G, k=6))

    # Number of k-shells
    # The k-shell is the subgraph induced by nodes with core number k. 
    # That is, nodes in the k-core that are not in the k+1-core
    
    # Number of 2-shells
    dictSums["num_2shells"] = len( nx.k_shell(G, 2) )
    
    # Number of 3-shells
    dictSums["num_3shells"] = len( nx.k_shell(G, 3) )
    
    # Number of 4-shells
    dictSums["num_4shells"] = len( nx.k_shell(G, 4) )
    
    # Number of 5-shells
    dictSums["num_5shells"] = len( nx.k_shell(G, 5) )
    
    # Number of 6-shells
    dictSums["num_6shells"] = len( nx.k_shell(G, 6) )
    
    
    listOfCliques = list(nx.enumerate_all_cliques(G))
    
    # Number of 4-cliques
    n4Clique = 0
    for li in listOfCliques:
        if len(li)==4:
            n4Clique += 1
    dictSums["num_4cliques"] = n4Clique

    # Number of 5-cliques
    n5Clique = 0
    for li in listOfCliques:
        if len(li)==5:
            n5Clique += 1
    dictSums["num_5cliques"] = n5Clique


    # Number of shortest path of size k
    listOfPLength = list(nx.shortest_path_length(G))
    
    # when k = 3
    n3Paths = 0
    for node_idx in range(G.number_of_nodes()):
        tmp = list( listOfPLength[node_idx][1].values() )
        n3Paths += tmp.count(3)
    dictSums["num_shortest_3paths"] = n3Paths/2

    # when k = 4
    n4Paths = 0
    for node_idx in range(G.number_of_nodes()):
        tmp = list( listOfPLength[node_idx][1].values() )
        n4Paths += tmp.count(4)
    dictSums["num_shortest_4paths"] = n4Paths/2

    # when k = 5
    n5Paths = 0
    for node_idx in range(G.number_of_nodes()):
        tmp = list( listOfPLength[node_idx][1].values() )
        n5Paths += tmp.count(5)
    dictSums["num_shortest_5paths"] = n5Paths/2
    
    # when k = 6
    n6Paths = 0
    for node_idx in range(G.number_of_nodes()):
        tmp = list( listOfPLength[node_idx][1].values() )
        n6Paths += tmp.count(6)
    dictSums["num_shortest_6paths"] = n6Paths/2
   
    # Size of the minimum (weight) node dominating set:
    # A subset of nodes where each node not in the subset has for direct 
    # neighbor a node of the dominating set.
    T = approximation.min_weighted_dominating_set(G)
    dictSums["size_min_node_dom_set"] = len(T)
    
    # Idem but with the edge dominating set
    T = approximation.min_edge_dominating_set(G) 
    dictSums["size_min_edge_dom_set"] = 2*len(T) # times 2 to have a number of nodes
    
    # Estrata index : sum_i^n exp(lambda_i)
    # with n the number of nodes, lamda_i the i-th eigen value of the adjacency matrix of G
    dictSums["Estrata_index"] = nx.estrada_index(G)

    # Eigenvector centrality
    # For each node, it is the average eigenvalue centrality of its neighbors,
    # where centrality of node i is taken as the i-th coordinate of x
    # such that Ax = lambda*x (for the maximal eigen value)
    
    # Averaged
    dictSums["avg_eigenvec_centrality"] = np.mean( list( nx.eigenvector_centrality_numpy(G).values() ) )

    # Maximum
    dictSums["max_eigenvec_centrality"] = max( list( nx.eigenvector_centrality_numpy(G).values() ) )
        
    return dictSums



def compute_indexed_summaries_undirected(G, sel_sum_names):
    """ Compute network features based on the indices of selected summaries (order based on the 46 base summaries above).
    
    This function evaluates only the selected summary statistics based on their
    indices. This function should be used after identifying relevant summaries.
    The indices refer to the order displayed in the function 
    compute_many_summaries_undirected (that computes 46 summaries).
        
        Args:
            G (networkx.classes.graph.Graph):
                an undirected networkx graph.
            sel_sum_names (list):
                a list that contains the names of the summary statistics of
                interest (as computed in compute_many_summaries_undirected).
        
        Returns:
            dictSums (dict):
                a dictionary with the name of the summaries as keys and the
                summary statistic values as values.
                
    """
    
    dictSums = dict()   # To store the summary statistic values
                        
    # Extract the largest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G_lcc = G.subgraph(Gcc[0])

    set_sel_sum_names = set(sel_sum_names)

    # Number of edges
    if "num_edges" in set_sel_sum_names:    
        dictSums["num_edges"] = G.number_of_edges()
    
    # Number of connected components
    if "num_of_CC" in set_sel_sum_names:    
        dictSums["num_of_CC"] = nx.number_connected_components(G)
    
    # Number of nodes in the largest connected component
    if "num_nodes_LCC" in set_sel_sum_names:    
        dictSums["num_nodes_LCC"] = nx.number_of_nodes(G_lcc)

    # Number of edges in the largest connected component
    if "num_edges_LCC" in set_sel_sum_names:    
        dictSums["num_edges_LCC"] = G_lcc.number_of_edges()
    
    # Diameter of the largest connected component
    if "diameter_LCC" in set_sel_sum_names:    
        dictSums["diameter_LCC"] = nx.diameter(G_lcc)
    
    # Average geodesic distance (shortest path length in the LCC)
    if "avg_geodesic_dist_LCC" in set_sel_sum_names:    
        dictSums["avg_geodesic_dist_LCC"] = nx.average_shortest_path_length(G_lcc)
    
    # Average degree of the neighborhood of each node
    if "avg_deg_connectivity" in set_sel_sum_names:    
        dictSums["avg_deg_connectivity"] = np.mean(list(nx.average_degree_connectivity(G).values()))
    
    # Average degree of the neighbors of each node in the LCC
    if "avg_deg_connectivity_LCC" in set_sel_sum_names:    
        dictSums["avg_deg_connectivity_LCC"] = np.mean(list(nx.average_degree_connectivity(G_lcc).values()))

    # Recover the degree distribution
    degree_vals = list(dict(G.degree()).values())
    
    # Entropy of the degree distribution
    if "degree_entropy" in set_sel_sum_names:    
        dictSums["degree_entropy"] = ss.entropy(degree_vals)
    
    # Maximum degree
    if "degree_max" in set_sel_sum_names:    
        dictSums["degree_max"] = max(degree_vals)

    # Average degree
    if "degree_mean" in set_sel_sum_names:    
        dictSums["degree_mean"] = np.mean(degree_vals)

    # Median degree
    if "degree_median" in set_sel_sum_names:    
        dictSums["degree_median"] = np.median(degree_vals)

    # Standard deviation of the degree distribution
    if "degree_std" in set_sel_sum_names:    
        dictSums["degree_std"] = np.std(degree_vals)

    # Quantile 25%
    if "degree_q025" in set_sel_sum_names:    
        dictSums["degree_q025"] = np.quantile(degree_vals, 0.25)
      
    # Quantile 75%
    if "degree_q075" in set_sel_sum_names:    
        dictSums["degree_q075"] = np.quantile(degree_vals, 0.75)
        
    # Average global efficiency:
    # The efficiency of a pair of nodes in a graph is the multiplicative 
    # inverse of the shortest path distance between the nodes.
    # The average global efficiency of a graph is the average efficiency of 
    # all pairs of nodes.
    if "avg_global_efficiency" in set_sel_sum_names:    
        dictSums["avg_global_efficiency"] = nx.global_efficiency(G)

    # Average local efficiency
    # The local efficiency of a node in the graph is the average global 
    # efficiency of the subgraph induced by the neighbors of the node. 
    # The average local efficiency is the average of the 
    # local efficiencies of each node.
    if "avg_local_efficiency_LCC" in set_sel_sum_names:    
        dictSums["avg_local_efficiency_LCC"] = nx.local_efficiency(G_lcc)
    
    # Node connectivity
    # The node connectivity is equal to the minimum number of nodes that 
    # must be removed to disconnect G or render it trivial.
    # Only on the largest connected component here.
    if "node_connectivity_LCC" in set_sel_sum_names:    
        dictSums["node_connectivity_LCC"] = nx.node_connectivity(G_lcc)

    # Edge connectivity
    # The edge connectivity is equal to the minimum number of edges that 
    # must be removed to disconnect G or render it trivial.
    # Only on the largest connected component here.
    if "edge_connectivity_LCC" in set_sel_sum_names:
        dictSums["edge_connectivity_LCC"] = nx.edge_connectivity(G_lcc)
    
    # Graph transitivity
    # 3*times the number of triangles divided by the number of triades
    if "transitivity" in set_sel_sum_names:    
        dictSums["transitivity"] = nx.transitivity(G)
    
    # Number of triangles
    if "num_triangles" in set_sel_sum_names:    
        dictSums["num_triangles"] = np.sum( list( nx.triangles(G).values() ) )/3
    
    # Estimate of the average clustering coefficient of G:
    # Average local clustering coefficient, with local clustering coefficient
    # defined as C_i = (nbr of pairs of neighbors of i that are connected)/(nbr of pairs of neighbors of i)
    if "avg_clustering_coef" in set_sel_sum_names:    
        dictSums["avg_clustering_coef"] = nx.average_clustering(G)
      
    # Square clustering (averaged over nodes): 
    # the fraction of possible squares that exist at the node.
    
    # We average it over nodes
    if "square_clustering_mean" in set_sel_sum_names:    
        dictSums["square_clustering_mean"] = np.mean( list( nx.square_clustering(G).values() ) )
 
    # We compute the median
    if "square_clustering_median" in set_sel_sum_names:    
        dictSums["square_clustering_median"] = np.median( list( nx.square_clustering(G).values() ) )

    # We compute the standard deviation
    if "square_clustering_std" in set_sel_sum_names:    
        dictSums["square_clustering_std"] = np.std( list( nx.square_clustering(G).values() ) )
    
    # Number of 2-cores
    if "num_2cores" in set_sel_sum_names:    
        dictSums["num_2cores"] = len(nx.k_core(G, k=2))
    
    # Number of 3-cores
    if "num_3cores" in set_sel_sum_names:    
        dictSums["num_3cores"] = len(nx.k_core(G, k=3))
    
    # Number of 4-cores
    if "num_4cores" in set_sel_sum_names:    
        dictSums["num_4cores"] = len(nx.k_core(G, k=4))
    
    # Number of 5-cores
    if "num_5cores" in set_sel_sum_names:    
        dictSums["num_5cores"] = len(nx.k_core(G, k=5))

    # Number of 6-cores
    if "num_6cores" in set_sel_sum_names:    
        dictSums["num_6cores"] = len(nx.k_core(G, k=6))

    # Number of k-shells
    # The k-shell is the subgraph induced by nodes with core number k. 
    # That is, nodes in the k-core that are not in the k+1-core
    
    # Number of 2-shells
    if "num_2shells" in set_sel_sum_names:    
        dictSums["num_2shells"] = len( nx.k_shell(G, 2) )
    
    # Number of 3-shells
    if "num_3shells" in set_sel_sum_names:    
        dictSums["num_3shells"] = len( nx.k_shell(G, 3) )
    
    # Number of 4-shells
    if "num_4shells" in set_sel_sum_names:    
        dictSums["num_4shells"] = len( nx.k_shell(G, 4) )
    
    # Number of 5-shells
    if "num_5shells" in set_sel_sum_names:    
        dictSums["num_5shells"] = len( nx.k_shell(G, 5) )
    
    # Number of 6-shells
    if "num_6shells" in set_sel_sum_names:    
        dictSums["num_6shells"] = len( nx.k_shell(G, 6) )
    
    
    if len(set_sel_sum_names.intersection({"num_4cliques","num_5cliques"}))>0:    
        listOfCliques = list(nx.enumerate_all_cliques(G))
    
        if "num_4cliques" in set_sel_sum_names:    
    
            # Number of 4-cliques
            n4Clique = 0
            for li in listOfCliques:
                if len(li)==4:
                    n4Clique += 1
            dictSums["num_4cliques"] = n4Clique
    
        if "num_5cliques" in set_sel_sum_names:    
            # Number of 5-cliques
            n5Clique = 0
            for li in listOfCliques:
                if len(li)==5:
                    n5Clique += 1
            dictSums["num_5cliques"] = n5Clique

    if len(set_sel_sum_names.intersection({"num_shortest_3paths","num_shortest_4paths","num_shortest_5paths","num_shortest_6paths"}))>0: 
        # Number of shortest path of size k
        listOfPLength = list(nx.shortest_path_length(G))
        
        if "num_shortest_3paths" in set_sel_sum_names:
            # when k = 3
            n3Paths = 0
            for node_idx in range(G.number_of_nodes()):
                tmp = list( listOfPLength[node_idx][1].values() )
                n3Paths += tmp.count(3)
            dictSums["num_shortest_3paths"] = n3Paths/2
    
        if "num_shortest_4paths" in set_sel_sum_names:
            # when k = 4
            n4Paths = 0
            for node_idx in range(G.number_of_nodes()):
                tmp = list( listOfPLength[node_idx][1].values() )
                n4Paths += tmp.count(4)
            dictSums["num_shortest_4paths"] = n4Paths/2
    
        if "num_shortest_5paths" in set_sel_sum_names:
            # when k = 5
            n5Paths = 0
            for node_idx in range(G.number_of_nodes()):
                tmp = list( listOfPLength[node_idx][1].values() )
                n5Paths += tmp.count(5)
            dictSums["num_shortest_5paths"] = n5Paths/2
        
        if "num_shortest_6paths" in set_sel_sum_names:
            # when k = 6
            n6Paths = 0
            for node_idx in range(G.number_of_nodes()):
                tmp = list( listOfPLength[node_idx][1].values() )
                n6Paths += tmp.count(6)
            dictSums["num_shortest_6paths"] = n6Paths/2
   
    # Size of the minimum (weight) node dominating set:
    # A subset of nodes where each node not in the subset has for direct 
    # neighbor a node of the dominating set.
    if "size_min_node_dom_set" in set_sel_sum_names:
        T = approximation.min_weighted_dominating_set(G)
        dictSums["size_min_node_dom_set"] = len(T)
    
    # Idem but with the edge dominating set
    if "size_min_edge_dom_set" in set_sel_sum_names:
        T = approximation.min_edge_dominating_set(G) 
        dictSums["size_min_edge_dom_set"] = 2*len(T) # times 2 to have a number of nodes
    
    # Estrata index : sum_i^n exp(lambda_i)
    # with n the number of nodes, lamda_i the i-th eigen value of the adjacency matrix of G
    if "Estrata_index" in set_sel_sum_names:
        dictSums["Estrata_index"] = nx.estrada_index(G)

    # Eigenvector centrality
    # For each node, it is the average eigenvalue centrality of its neighbors,
    # where centrality of node i is taken as the i-th coordinate of x
    # such that Ax = lambda*x (for the maximal eigen value)
    
    # Averaged
    if "avg_eigenvec_centrality" in set_sel_sum_names:
        dictSums["avg_eigenvec_centrality"] = np.mean( list( nx.eigenvector_centrality_numpy(G).values() ) )

    # Maximum
    if "max_eigenvec_centrality" in set_sel_sum_names:
        dictSums["max_eigenvec_centrality"] = max( list( nx.eigenvector_centrality_numpy(G).values() ) )
        
    return dictSums