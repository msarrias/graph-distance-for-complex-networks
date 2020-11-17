import numpy as np
import networkx as nx
from scipy.integrate import quad
import scipy.linalg as la
from matplotlib import pyplot as plt
import random, time, math, json
from collections import Counter
from numpy.linalg import eigh

class SGD:
    
    def __init__(self, ref, solve, G_ref, G):
        
        self.G_ref = G_ref
        self.ref = ref
        self.n = len(G_ref)
        self.G = G
        self.keys = list(G.keys())
        self.replicates = len(G[self.keys[0]])
        self.eps = 1e-10
        self.solve = solve
        self.signs = [[1,1], [-1,1]]
    
    def adj_degree_matrices(self, graph):
        A = (nx.adjacency_matrix(graph)).todense()
        D = np.diag(np.asarray(sum(A))[0])
        return A, D
    
    def sort_eigenv(self, eigenvalues, eigenvectors):
        return sorted(zip(eigenvalues.real,
                          eigenvectors.T),
                      key=lambda x: x[0])
    
    def normalize_eigenv(self, eigenvcts):
        num = eigenvcts - min(eigenvcts)
        dem = max(eigenvcts) - min(eigenvcts)
        return (num / dem)
    
    def emp_cdf_distance(self, vri, vrj):
        u = sorted(list(set([*vri , *vrj])))
        vri_counter = Counter(vri)
        vrj_counter = Counter(vrj)
        vri_count = vri_counter.get(u[0], 0)
        vrj_count = vrj_counter.get(u[0], 0)
        cdf_distance = 0
        Ni = len(vri)
        Nj = len(vrj)
        
        for idx, sorted_elt in enumerate(u[1:]):
            # adding rectangle area, L * W
            abs_dif = abs((vri_count / Ni) - (vrj_count / Nj))
            cdf_distance += (u[idx + 1] - u[idx]) * abs_dif
            vri_count += vri_counter.get(sorted_elt, 0)
            vrj_count += vrj_counter.get(sorted_elt, 0)
        return cdf_distance

    def eigenspectrum_G_ref(self):
        self.A_ref, self.D_ref = self.adj_degree_matrices(self.G_ref)
        
        if self.solve == "standard":
            eigenvs_ref, eigenvec_ref = la.eigh(self.D_ref - self.A_ref)
            self.sort_eigenv_ref = self.sort_eigenv(eigenvs_ref, 
                                                   eigenvec_ref)
        
        if self.solve == "generalized":
            eigenvs_ref, eigenvec_ref = la.eig(self.D_ref - self.A_ref, 
                                             self.D_ref)
            self.sort_eigenv_ref = self.sort_eigenv(eigenvs_ref,
                                                   eigenvec_ref)
        print(f'Eigen-decomposition of reference model with ' \
              f'ref param : {self.ref} completed.')
    
    def reset_G_ref(self, new_ref, new_G_ref):
        self.G_ref = new_G_ref
        self.ref = new_ref
        
    def eigenspectrum_G(self):
        tic = time.time() 
        self.sort_eigenv_dic = {}
        self.A_dic_G = {}
        
        if self.solve == "standard":
            for key in self.keys:
                eigenv_dic = {}
                A_dic = {}
                for j in range(self.replicates):
                    A_dic[j], D = self.adj_degree_matrices(self.G[key][j])
                    eigenvs, eigenvcts = la.eigh(D - A_dic[j])
                    eigenv_dic[j] = self.sort_eigenv(eigenvs, eigenvcts) 
                self.A_dic_G[key] =  A_dic
                self.sort_eigenv_dic[key] = eigenv_dic
        
        if self.solve == "generalized":
             for key in self.keys:
                eigenv_dic = {}
                A_dic = {}
                for j in range(self.replicates):
                    A_dic[j], D = self.adj_degree_matrices(self.G[key][j])
                    eigenvs, eigenvcts = la.eig(D - A_dic[j], D)
                    eigenv_dic[j] = self.sort_eigenv(eigenvs, eigenvcts)
                self.A_dic_G[key] =  A_dic
                self.sort_eigenv_dic[key] = eigenv_dic
            
        print(f'Eigen-decomposition of the {self.replicates * len(self.keys)}' \
              f' models completed.')
        print(f'took {np.round((time.time()  - tic) / 60, 3)} minutes.')
     
    def sgd(self, r, G2_dic):
        temp_integral = []
        for sign in self.signs:
            eigvi = sign[0] * self.sort_eigenv_ref[r][1]
            eigvj = sign[1] * G2_dic[r][1]
            vri = sorted(self.normalize_eigenv(eigvi))
            vrj = sorted(self.normalize_eigenv(eigvj))
            temp_integral.append(self.emp_cdf_distance(vri, vrj)) 
        return min(temp_integral)

    def fit_SGD(self):
        tic = time.time()
        print(f'The spectral distance between G({self.ref}) and G(~) is an average  ' \
              f'of the SGD between G({self.ref}) and the {self.replicates} replicate models' \
              f' for the {len(self.keys)} generations.')
        print()
        self.rep_sgd = {}
        self.time = []
    
        for idx, key in enumerate(self.keys):
            tic_ = time.time() 
            replicate_distance = []
            for rep in range(self.replicates):
                G2_dic  = self.sort_eigenv_dic[key][rep]
                integral_list = []
                for r in range(1, self.n):
                    integral_list.append(self.sgd(r, G2_dic))
                replicate_distance.append(sum(integral_list)/(self.n - 1))
            self.rep_sgd[key] = replicate_distance
            tac_ = time.time()
            time_ = (tac_ - tic_) / 60
            self.time.append(time_)
            print(f'd(G({np.round(self.ref,3)}), G({np.round(key,3)})): {np.mean(self.rep_sgd[key])}, ' \
                  f'took: {np.round(time_, 4)} minutes') 
        tac = time.time()
        print()
        print(f'Process took {(tac-tic)/60} minutes with an average of {np.mean(self.time)} per generation')