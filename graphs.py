#https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/social.html#karate_club_graph

import networkx as nx
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.integrate import quad
import scipy.linalg as la
import networkx as nx
import seaborn as sns
import pandas as pd

def sgd_matrix(reign_lists):
    N_graphs = len(reign_lists)
    graph_dist_matx = np.zeros((N_graphs,len(reign_lists)))
    for i in range(N_graphs):
        for j in range(i, N_graphs):
            _, graph_dist = graph_pair_distance(reign_lists[i],
                                                reign_lists[j])
            graph_dist_matx[i,j] = graph_dist
            graph_dist_matx[j,i] = graph_dist
    return graph_dist_matx

def cdf(N,y,v):
    p = np.zeros((np.shape(y)))
    for i in range(len(y)):
        tr = y[i] - v
        p[i] =  len(tr[tr>=0])/N
    return(p)

def cdf_(N,y,v):
    tr = y - v
    p =  len(tr[tr>=0])/N
    return(p)

def normalize_eigenv(eigenvc):
    return((eigenvc - min(eigenvc)
           )/(max(eigenvc) - min(eigenvc)))


def integrand(y, N_i, N_j, v_ri, v_rj):
    return np.abs(cdf_(N_i,y,v_ri) - cdf_(N_j,y,v_rj))

def cdf_distance(eig_G1, eig_G2, r):
    N_i= len(eig_G1)
    N_j= len(eig_G2)
    v_ri = normalize_eigenv(sorted(eig_G1[r][1],
                                   key=lambda x: x))
    v_rj = normalize_eigenv(sorted(eig_G2[r][1],
                                   key=lambda x: x))
    return (quad(integrand, -np.inf, np.inf, 
                 epsabs = 1e-4, limit = 500, 
                 args=(N_i,N_j,v_ri,v_rj))[0])
    
    
def graph_pair_distance(eig_G1,eig_G2):
    Mij = min(len(eig_G1), len(eig_G2))
    eigenvect_dist = np.zeros((Mij-1))
    for i in range(1,Mij):
        eigenvect_dist[i-1] = cdf_distance(eig_G1,
                                           eig_G2, i)
    return(eigenvect_dist,
           sum(eigenvect_dist)/(Mij-1))  

def taro_graph():
    # Create the set of all members
    all_members = set(range(22))

    G = nx.Graph()
    G.add_nodes_from(all_members)
    G.name = "SCHWIMMER TARO EXCHANGE"

    tarodata = """\
0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0
0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0
0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 1 1 0
0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 1 0 0 0
0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1
0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1
0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1
0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0
0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0"""

    for row, line in enumerate(tarodata.split('\n')):
        thisrow = [int(b) for b in line.split()]
        for col, entry in enumerate(thisrow):
            if entry == 1:
                G.add_edge(row, col)
    return G

def savage_plot(graphs_list):
    f, axs = plt.subplots(2,3,figsize=(20,15))
    nx.draw_networkx(graphs_list[0], ax=axs[0,0], 
                     node_color='lightgrey')
    axs[0,0].axis('off')
    axs[0,0].set_title("2D grid")
    nx.draw_networkx(graphs_list[1], ax=axs[0,1], 
                     node_color='lightgreen')
    axs[0,1].axis('off')
    axs[0,1].set_title("Dolphins")
    nx.draw_networkx(graphs_list[2], ax=axs[0,2], 
                     node_color='lightblue')
    axs[0,2].axis('off')
    axs[0,2].set_title("Taro")
    nx.draw_networkx(graphs_list[3], ax=axs[1,0],
                     node_color='lightpink')
    axs[1,0].axis('off')
    axs[1,0].set_title("Karate")
    nx.draw_networkx(graphs_list[4], ax=axs[1,1],
                     node_color='lightyellow')
    axs[1,1].axis('off')
    axs[1,1].set_title("Southern women")
    nx.draw_networkx(graphs_list[5], ax=axs[1,2], 
                     node_color='lightgrey')
    axs[1,2].axis('off')
    axs[1,2].set_title("Florentine Families")
    plt.show()
    