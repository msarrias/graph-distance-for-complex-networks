class Graph:
    def __init__(self, nnodes, weight=1):
        self.nnodes = nnodes
        self.init_nodes()
        self.weighted_edges = []
        self.edges = []
        self.V = 0
        self.weight = weight
        
    def init_nodes(self):
        self.graph = {int_val: dict()
                      for int_val in range(1, self.nnodes+1)}
        
    def add_edge(self, node1, node2, weight=1):
        if (node1, node2) not in self.edges and node1 != node2:
            self.V +=1
            self.graph[node1][node2] = weight
            self.graph[node2][node1] = weight
            self.edges.append((node1,node2))
            self.edges.append((node2, node1)) 
            if node1 < node2 : 
                self.weighted_edges.append((node1-1, node2-1, weight))
            else:
                self.weighted_edges.append((node2-1, node1-1, weight))
        
    def add_nn_edges(self, k, weight):
        m = int(k/2)
        #Set lattice vertices
        padded_list = self.nodes() * 3
        self.lattice_vertices = {self.nodes()[i]: 
                                 sorted(padded_list[j - m : j][:: -1]
                                        + padded_list[j + 1: j + m + 1]) 
                for i, j in zip(range(self.nnodes),
                                range(self.nnodes, 2 * self.nnodes))}
        #Add lattice vertices to graph
        for node in self.graph:
            self.graph[node] = {neighbor: self.weight
                                for neighbor in
                                self.lattice_vertices[node]}
        
        for key, value in self.lattice_vertices.items():
            for item in value:
                self.edges.append((key, item))
                if key < item:
                    self.weighted_edges.append((key-1, item-1, weight))
        
    def number_of_vertices(self):
        return len(self.weighted_edges)

    def find_edge(self, node1, node2):
        try:
            return self.graph[node1][node2]
        except KeyError:
            return None
                                                  
    def find_isolated_nodes(self):
        if any(edges != {} for edges in self.graph.values()):
            self.isolated_nodes = []
        else:
            self.isolated_nodes = []
            for node, edges in self.graph.items():
                if edges == {}:
                    self.isolated_nodes.append(node)
    
    def remove_edge(self, node1, node2):
        if self.find_edge(node1, node2) != None:
            del self.graph[node1][node2]
            del self.graph[node2][node1]
            self.edges.remove((node1, node2))
            self.edges.remove((node2, node1))
            if node1 < node2: 
                self.weighted_edges.remove((node1-1, node2-1, self.weight))
            else: 
                self.weighted_edges.remove((node2-1, node1-1, self.weight))
            
            
    def nodes(self):
        return list(self.graph.keys())
    
    def node_edges(self, node):
        nd_edges = []
        for edge in self.graph[node]:
            if self.graph[node][edge] != {}:
                if (node, edge) not in nd_edges:
                    nd_edges.append((node, edge))
                    nd_edges.append((edge, node))
        return(nd_edges)
            
    
    def adjacency_matrix(self):
        A = np.zeros((self.nnodes, self.nnodes))
        for i in self.weighted_edges: 
            r, c, w = i
            A[r][c] = w
            A[c][r] = A[r][c]
        
        return A
   
    def is_connected(self):
        self.find_isolated_nodes()
        if any(self.isolated_nodes):
            return False
        
        v0 = self.nodes()[0]
        queue = []
        visited = []
        queue.append(v0)
        
        while (queue):
            v = queue[0]
            visited.append(v)
            queue.remove(queue[0])
            for node in self.graph[v]:
                if node not in visited and node not in queue:
                    queue.append(node)

        if set(visited) == set(self.nodes()):
            return True
        else:
            return False
