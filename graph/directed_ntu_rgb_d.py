from typing import Tuple, List
from collections import defaultdict

import numpy as np
# import scipy as sp
from scipy import special

# For NTU RGB+D, assume node 21 (centre of chest)
# is the "centre of gravity" mentioned in the paper

num_nodes = 25
epsilon = 1e-6

# Directed edges: (source, target), see
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf
# for node IDs, and reduce index to 0-based
directed_edges = [(i-1, j-1) for i, j in [
    (1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
    (6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
    (10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
    (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
    (21, 2), (21, 3), (21, 5), (21, 9)
]]

self_loops = [(i, i) for i in range(num_nodes)]

# outgoing_edges = [(i-1, j-1) for i, j in [
#     (1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
#     (6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
#     (10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
#     (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
#     (21, 2), (21, 3), (21, 5), (21, 9)
# ]]

# incoming_edges = [(target, source) for source, target in outgoing_edges]


def build_digraph_adj_list(edges: List[Tuple]) -> np.ndarray:
    graph = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
    return graph


def normalize_incidence_matrix(im: np.ndarray) -> np.ndarray:
    # NOTE: The paper assumes that the Incidence matrix is square,
    # so that the normalized form A @ (D ** -1) is viable.
    # However, if the incidence matrix is non-square, then
    # the above normalization won't work.
    # For now, move the term (D ** -1) to the front
    degree_mat = im.sum(-1) * np.eye(len(im))

    # Since all nodes should have at least some edge, degree matrix is invertible
    inv_degree_mat = np.linalg.inv(degree_mat)
    return (inv_degree_mat @ im) + epsilon


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    # Consider all possible edges
    max_edges = int(special.comb(num_nodes, 2))
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1
        target_graph[target_node, edge_id] = 1

    source_graph = normalize_incidence_matrix(source_graph)
    target_graph = normalize_incidence_matrix(target_graph)
    return source_graph, target_graph


def build_digraph_adj_matrix(edges: List[Tuple]) -> np.ndarray:
    graph = np.zeros((num_nodes, num_nodes), dtype='float32')
    for edge in edges:
        graph[edge] = 1
    return graph


class Graph:
    def __init__(self):
        super().__init__()
        self.num_nodes = num_nodes
        self.edges = directed_edges + self_loops
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.edges)

        # self.A = build_digraph_adj_matrix(outgoing_edges)
        # self.in_edges = incoming_edges
        # self.out_edges = outgoing_edges
        # self.num_nodes = num_nodes
        # self.self_loops = [(i, i) for i in range(self.num_nodes)]


# TODO:
'''
Figure out whether self loop should be added inside the graph
Figure out whether edges need to be included in this graph class

Figure out incidence matrix size
Need to change how data is read in; the 2nd stream is now temporal, so both edge and nodes need to be procedded when reading in
'''

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    graph = Graph()
    source_M = graph.source_M
    target_M = graph.target_M
    plt.imshow(source_M, cmap='gray')
    plt.show()
    plt.imshow(target_M, cmap='gray')
    plt.show()
    print(source_M)
    # print()