# -*- coding: utf-8 -*-
"""
Implementation of mechanisms for model of mixture of mechanisms.
"""

import numpy as np
from mechanistic_net_abc.utility import _find_paths

##### Basic growth

def node_addition(G, m):
    """ Function to add one node, without edges
    
    We here implement the mechanism that adds m node to the network, without
    edges.

    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph to modify.
        m (int):
            the number of nodes to add to the existing network.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
    
    """
    
    # Add the m new nodes
    nodes_to_add = [max(list(G.nodes)) + i for i in range(1,m+1)]
    G.add_nodes_from(nodes_to_add)


##### Mechanims of preferential attachment

def preferential_attachment_growth(G, m, degree_aug=0, seed=None):
    """ Implementation of one step of growth with preferential attachment.
    
    We here implement the mechanism of growth with preferential attachment, 
    where a new node is added and establishes a connection with m existing
    nodes, with probability of attachment proportional to the degree k
    of the existing nodes + k_0.
    
    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph for which a new node will be added and the mechanism
            will be applied on.
        m (int):
            the number of edges to attach from the new node to existing nodes.
        degree_aug (float):
            the degree of all nodes is artificially increased by degree_aug.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for node sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
                
    """

    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random    

    if G.number_of_nodes() < m:
        raise ValueError('The number of nodes in the seed graph cannot be lower than the parameter m.')

    # Recover the dict of degree for G
    degree_dict_G = dict(G.degree)

    # Normalize the degree vector between 0 and 1 (note: each edge is counted twice)
    aug_degree  = (np.array(list(degree_dict_G.values())) + degree_aug)
    proba_degree = aug_degree / aug_degree.sum()
    
    # Sample the m nodes that will be connected to the new node, based on
    # degree of each existing nodes (sampling without replacement)
    # If there is no connected nodes, we use the random_attachment_growth
    if G.number_of_edges() > 0:
        target_nodes = random_seed.choice(list(degree_dict_G.keys()), size = m,
                                          replace = False, p = proba_degree)
        # Add the new node and corresponding new edges
        G.add_edges_from(zip([max(list(G.nodes)) + 1] * m, target_nodes))
    else:
        random_attachment_growth(G, m, seed=seed)


def neg_preferential_attachment_growth(G, m, degree_aug=0, seed=None):
    """ Implementation of one step of growth with negative preferential attachment.
    
    We here implement the mechanism of growth with preferential attachment, 
    where a new node is added and establishes a connection with m existing
    nodes, with probability of attachment inversely proportional to (the degree 
    of the existing nodes + degree_aug). 
    
    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph for which a new node will be added and the mechanism
            will be applied on.
        m (int):
            the number of edges to attach from the new node to existing nodes.
        degree_aug (float):
            the degree of all nodes is artificially increased by degree_aug.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for node sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
                
    """
    
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
        
    if G.number_of_nodes() < m:
        raise ValueError('The number of nodes in the seed graph cannot be lower than the parameter m.')
        
    # Recover the dict of degree for G
    degree_dict_G = dict(G.degree)

    # Normalize the degree vector between 0 and 1 (note: each edge is counted twice)
    if G.number_of_edges() > 0:
        inv_degree = 1 / ( np.array(list(degree_dict_G.values())) + degree_aug )
        proba_degree = inv_degree / inv_degree.sum()
    
    # Sample the m nodes that will be connected to the new node, based on
    # degree of each existing nodes (sampling without replacement)
    # if seed is not None:
    #     random.seed(seed)
    # If there is no connected nodes, we use the random_attachment_growth
    if G.number_of_edges() > 0:
        target_nodes = random_seed.choice(list(degree_dict_G.keys()), size = m,
                                          replace = False, p = proba_degree)
        # Add the new node and corresponding new edges
        G.add_edges_from(zip([max(list(G.nodes)) + 1] * m, target_nodes))
    #else:
    #   random_attachment_growth(G, m, seed=seed)


def random_attachment_growth(G, m, seed=None):
    """ Implementation of one step of growth with random attachment.
    
    We here implement the mechanism of growth with random attachment, 
    where a new node is added and establishes a connection with m existing
    nodes, with uniform probability.
    
    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph for which a new node will be added and the mechanism
            will be applied on.
        m (int):
            the number of edges to attach from the new node to existing nodes.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for node sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
                
    """
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
        
    if G.number_of_nodes() < m:
        raise ValueError('The number of nodes in the seed graph cannot be lower than the parameter m.')

    # Recover the list of nodes
    node_list = list(G.nodes())
    
    # Sample the m nodes that will be connected to the new node,
    # selected randomly (sampling without replacement)
    target_nodes = random_seed.choice(node_list, size = m,
                                      replace = False)
    # Add the new node and corresponding new edges
    G.add_edges_from(zip([max(list(G.nodes)) + 1] * m, target_nodes))
    
    
##### Mechanisms of random deletion (node or edges)

def random_node_deletion(G, m, seed=None):
    """ Implementation of one step of random node deletion.
    
    We here implement the mechanism of random node deletion, 
    where m nodes of the network G are randomly deleted with uniform probability.
    
    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph for which m nodes will be deleted.
        m (int):
            the number of nodes to delete.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for node sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
                
    """
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
        
    if G.number_of_nodes() < m:
        raise ValueError('The number of nodes in the seed graph cannot be lower than the number of nodes to delete m.')

    # Recover the list of nodes
    node_list = G.nodes()
    
    # If there is no nodes, do nothing
    if len(node_list) > 0:
        # Sample the m nodes that will be deleted,
        # selected randomly (sampling without replacement)
        target_nodes = random_seed.choice(node_list, size = m,
                                   replace = False)
        # Delete the selected nodes and their edges
        for node in target_nodes:
            G.remove_node(node)


def random_edge_deletion(G, m, seed=None):
    """ Implementation of one step of random edge deletion.
    
    We here implement the mechanism of random edge deletion, 
    where m edges of the network G are randomly deleted with uniform probability.
    
    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph for which m edges will be deleted.
        m (int):
            the number of edges to delete.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for edge sampling. None by default.        
            
    Returns:
        None:
            this function directly modifies the existing graph G.
                
    """

    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
        
    if G.number_of_edges() < m:
        raise ValueError('The number of edges in the graph cannot be lower than the number of edges to delete m.')

    # Recover the list of nodes
    edge_list = list(G.edges())
    
    # if there is no edges, do nothing
    if len(edge_list) > 0:
        idx_list = range(len(edge_list))    
        # Sample the m nodes that will be deleted,
        # selected randomly (sampling without replacement)
        target_idx = random_seed.choice(idx_list, size = m,
                                        replace = False)
        # Delete the selected edges
        for idx in target_idx:
            G.remove_edge(*edge_list[idx])


##### Rewiring mechanisms

def random_edge_rewiring(G, seed=None):
    """ Implementation of one step of random edge rewiring

    This mechanism randomly chooses a node, and drops one of its existing
    edges randomly, to then choose another node (with probability p?). 
    The rewiring excludes the nodes that are already connected to the node 
    subject to rewiring.

    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph on which the mechanism will be applied.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for edge sampling. None by default.

    Returns:
        None:
            this function directly modifies the existing graph G.
        
    """
    
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
    
    # Randomly select a node
    node_list = list(G.nodes())
    target_node = random_seed.choice(node_list)

    # If the node is conected to another one, rewire
    if G.degree(target_node) > 0:
        # Select and delete an edge
        neighbor_node = random_seed.choice(list(G.neighbors(target_node)))
        G.remove_edge(target_node,
                      neighbor_node)
        # We cannot select target_node
        node_list.remove(target_node)
        # and none of its current neighbors
        for neighbor in list(G.neighbors(target_node)):
            node_list.remove(neighbor)
        # Rewire target_node to another possible node
        G.add_edge(target_node, np.random.choice(node_list))
        

def triadic_closure(G, seed=None):
    """ Implementation of one step of triadic closure

    This mechanism randomly selects a two-edge path (A-B-C).
    If there is no edge between A and C, one of these nodes randomly loses one 
    of its edges (not the one connecting to B), and the edge A-C is formed to
    create a triangle.

    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph on which the mechanism will be applied.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for edge sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
        
    """

    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random

    ### List all the two-edge paths
    list_two_edge_paths = []
    for node in G.nodes():
        list_two_edge_paths += _find_paths(G, node, 2, excludeSet = None)
    
    if len(list_two_edge_paths) > 0:
        idx_list = range(len(list_two_edge_paths))
        # Sample the triad that we might want to close
        target_idx = random_seed.choice(idx_list)
        target_triad = list_two_edge_paths[target_idx]
        
        # If A-C does not aready exist:
        if not G.has_edge(target_triad[0], target_triad[2]):
            # Choose the node A or C with equal probability
            if random_seed.binomial(1, 0.5):
                A_or_C = 0 # work on A
            else:
                A_or_C = 2 # work on C
            # Remove node B of the possible selected neighboring nodes 
            neighbors = list(G.neighbors(target_triad[A_or_C]))
            neighbors.remove(target_triad[1])
            if len(neighbors) > 0:
                # Select the neighboring not to delete the edge with (not B)
                neighbor_node = random_seed.choice(neighbors)
                G.remove_edge(target_triad[A_or_C], neighbor_node)
                # Create the edge A-C
                G.add_edge(target_triad[0], target_triad[2])
                
### Triadic closure without rewiring process

def triadic_closure_without_rewiring(G, seed=None):
    """ Implementation of one step of triadic closure without rewiring

    This mechanism randomly selects a two-edge path (A-B-C).
    If there is no edge between A and C, create the edge A-C to create a triangle.

    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph on which the mechanism will be applied.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for edge sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
        
    """
    
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
            
    ### List all the two-edge paths
    list_two_edge_paths = []
    for node in G.nodes():
        list_two_edge_paths += _find_paths(G, node, 2, excludeSet = None)
    
    if len(list_two_edge_paths) > 0:
        idx_list = range(len(list_two_edge_paths))
        # Sample the triad that we might want to close
        target_idx = random_seed.choice(idx_list)
        target_triad = list_two_edge_paths[target_idx]
        
        # If A-C does not aready exist:
        if not G.has_edge(target_triad[0], target_triad[2]):
            G.add_edge(target_triad[0], target_triad[2])
            

# Triangle formation with an existing edge selected at random

def triangle_formation_node_addition(G, seed=None):
    """ Implementation of one step of triangle formation on a random edge

    This mechanism randomly selects an edge (A-B).
    A new node C is added and the triangle (A-B),(B-C), (C-A) forms.

    Args:
        G (networkx.classes.graph.Graph):
            a networkx graph on which the mechanism will be applied.
        seed (numpy.random.mtrand.RandomState):
            the numpy RandomState to use for edge sampling. None by default.
            
    Returns:
        None:
            this function directly modifies the existing graph G.
        
    """
    
    if seed is not None:
        random_seed = seed
    else:
        random_seed = np.random
            
    ### List all the edges
    edge_list = list(G.edges())
    
    # if there is no edges, do nothing
    if len(edge_list) > 0:
        idx_list = range(len(edge_list))    
        # Select the edge index to form triangle
        idx_target_edge = random_seed.choice(idx_list, size = 1)
        new_node = max(list(G.nodes)) + 1
        # Add the two new edges
        G.add_edge(new_node, edge_list[idx_target_edge[0]][0])
        G.add_edge(new_node, edge_list[idx_target_edge[0]][1])