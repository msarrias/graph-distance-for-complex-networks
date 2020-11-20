import numpy as np
import random
from Graph import Graph

class watts_strogatz_graph:
    
    def __init__(self, N, k, p, weight = 1):
        if N*k%2 != 0:
            raise Exception('k must be even :-)')
        if N < k+1:
            raise Exception(r'set n st: n >= k + 1')
        self.k = k
        self.N = N
        self.p = p
        self.weight = weight
        self.WS = Graph(self.N)
        self.WS.add_nn_edges(self.k, self.weight)
        self.rewire_WS_model()
        
    def rewire_WS_model(self):
        for node, links in self.WS.graph.items():
            keys_to_iterate = list(links.keys())
            for link in keys_to_iterate:
                eps = random.uniform(0,1)
                if eps < self.p:
                    new_edge = random.choice([n for n in self.WS.nodes()
                                             if n not in [node, link]
                                             and self.WS.find_edge(node, n) == None])
                    self.WS.remove_edge(node, link)
                    self.WS.add_edge(node, new_edge)
 
    def clustering_coefficient(self):
        C = 0
        for node, neighbors_node in self.WS.graph.items():
            max_numb_neighbor_edges = (len(neighbors_node) * (len(neighbors_node) - 1)) / 2.0
            if max_numb_neighbor_edges <= 0:
                continue
            neighbor_edges = []
            # find which  neighbors of "node" are connected to every other neighbor of "node".
            for n_node in neighbors_node:
                #go through neighbors of neighbor of "node"
                for nn_node in self.WS.graph[n_node]:
                    if nn_node in neighbors_node:
                        if (n_node, nn_node) and ((nn_node, n_node)) not in neighbor_edges:
                            if (n_node, nn_node) and (nn_node, n_node) not in self.WS.node_edges(node):
                                neighbor_edges.append((n_node, nn_node))
                                neighbor_edges.append((nn_node, n_node))
                                
            numb_neighbor_edges = len(neighbor_edges)/2.0
            C += numb_neighbor_edges / max_numb_neighbor_edges
        return C / len(self.WS.graph.keys())