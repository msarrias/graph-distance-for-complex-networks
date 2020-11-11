import numpy as np
import networkx as nx
from scipy.integrate import quad
import scipy.linalg as la
from matplotlib import pyplot as plt
import random, time, math, json
from collections import Counter

class SGD_WSmodels:
    
    def __init__(self, p0, k, solve, G_p0, G_p):
        
        self.G_p0 = G_p0
        self.p0 = p0
        self.n = len(G_p0)
        self.G_p = G_p
        self.k = k
        self.p = list(G_p.keys())
        self.replicates = len(G_p[self.p[0]])
        self.eps = 1e-10
        self.solve = solve
        self.signs = [-1, 1]
    
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

    def eigenspectrum_G_p0(self):
        self.A_p0, self.D_p0 = self.adj_degree_matrices(self.G_p0)
        
        if self.solve == "standard":
            eigenvs_p0, eigenvec_p0 = la.eig(self.D_p0 - self.A_p0)
            self.sort_eigenv_p0 = self.sort_eigenv(eigenvs_p0, 
                                                   eigenvec_p0)
        
        if self.solve == "generalized":
            eigenvs_p0, eigenvec_p0 = la.eig(self.D_p0 - self.A_p0, 
                                             self.D_p0)
            self.sort_eigenv_p0 = self.sort_eigenv(eigenvs_p0,
                                                   eigenvec_p0)
        print(f'Eigen-decomposition of reference model with ' \
              f'rewiring prob p : {self.p0} completed.')
    
    def reset_G_p0(self, new_p0, new_G_p0):
        self.G_p0 = new_G_p0
        self.p0 = new_p0
        
    def eigenspectrum_G_p(self):
        tic = time.time() 
        self.sort_eigenv_dic = {}
        self.A_dic_G_p = {}
        
        if self.solve == "standard":
            for ps in self.p:
                eigenv_dic = {}
                A_dic_p = {}
                for j in range(self.replicates):
                    A_dic_p[j], D = self.adj_degree_matrices(self.G_p[ps][j])
                    eigenvs, eigenvcts = la.eig(D - A_dic_p[j])
                    eigenv_dic[j] = self.sort_eigenv(eigenvs, eigenvcts) 
                self.A_dic_G_p[ps] =  A_dic_p
                self.sort_eigenv_dic[ps] = eigenv_dic
        
        if self.solve == "generalized":
             for ps in self.p:
                eigenv_dic = {}
                A_dic_p = {}
                for j in range(self.replicates):
                    A_dic_p[j], D = self.adj_degree_matrices(self.G_p[ps][j])
                    eigenvs, eigenvcts = la.eig(D - A_dic_p[j], D)
                    eigenv_dic[j] = self.sort_eigenv(eigenvs, eigenvcts)
                self.A_dic_G_p[ps] =  A_dic_p
                self.sort_eigenv_dic[ps] = eigenv_dic
            
        print(f'Eigen-decomposition of the {self.replicates * len(self.p)}' \
              f' WS models completed.')
        print(f'took {np.round((time.time()  - tic) / 60, 3)} minutes.')
     
    def sgd(self, r, G2_dic_p):
        temp_integral = []
        for sign_s in self.signs:
            for sign_l in self.signs:
                eigvi = sign_s * self.sort_eigenv_p0[r][1]
                eigvj = sign_l * G2_dic_p[r][1]
                vri = sorted(self.normalize_eigenv(eigvi))
                vrj = sorted(self.normalize_eigenv(eigvj))
                temp_integral.append(self.emp_cdf_distance(vri, vrj)) 
        return min(temp_integral)

    def fit_SGD(self):
        tic = time.time()
        print(f'The spectral distance between G({self.p0}) and G(p) is an average  ' \
              f'of the SGD between G({self.p0}) and the {self.replicates} replicate models' \
              f' generated per each of the {len(self.p)}' \
              f' rewiring probability.')
        print()
        self.rep_sgd = {}
        self.time_p = []
    
        for idx, rp in enumerate(self.p):
            tic_p = time.time() 
            replicate_distance = []
            for rep in range(self.replicates):
                G2_dic_p  = self.sort_eigenv_dic[rp][rep]
                integral_list = []
                for r in range(1, self.n):
                    integral_list.append(self.sgd(r, G2_dic_p))
                replicate_distance.append(sum(integral_list)/(self.n - 1))
            self.rep_sgd[rp] = replicate_distance
            tac_p = time.time()
            time_rp = (tac_p - tic_p) / 60
            self.time_p.append(time_rp)
            print(f'd(G({self.p0}), G({rp})): {np.mean(self.rep_sgd[rp])}, ' \
                  f'took: {np.round(time_rp, 4)} minutes') 
        tac = time.time()
        print()
        print(f'Process took {(tac-tic)/60} minutes with an average of {np.mean(self.time_p)} per p')