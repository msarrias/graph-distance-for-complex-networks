import numpy as np
import random
from Graph import Graph

class erdos_renyi_graph:
    def __init__(self, N, p, weight = 1, seed = False):
        if seed: random.seed(seed)       
        self.N = N
        self.p = p
        self.weight = weight
        self.ER = Graph(self.N)
        self.ER.init_nodes()
        self.generate_ER_graph()
        
    def generate_ER_graph(self):
        for i in self.ER.nodes(): 
            for j in self.ER.nodes(): 
                if (i < j):  
                    eps = random.uniform(0,1)
                    if (eps < self.p): 
                        self.ER.add_edge(i, j, self.weight)
