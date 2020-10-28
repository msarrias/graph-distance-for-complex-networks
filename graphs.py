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


def cdf_(N, y, v):
    tr = y - v
    return len(tr[tr>=0]) / N

def cdf(N,y,v):
    p = np.zeros((np.shape(y)))
    for i in range(len(y)):
        tr = y[i] - v
        p[i] =  len(tr[tr>=0])/N
    return(p)

def normalize_eigenv(eigenvc):
    return(
        (eigenvc - min(eigenvc)) / (max(eigenvc) - min(eigenvc))
    )

def integrand_kd(y, N_i, N_j, v_ri, v_rj):
    return np.abs((np.sum(y - v_ri >= 0) / N_i) - (np.sum(y - v_rj >= 0) / N_j))

def graphs_kruglov_distance(eig_Gi, eig_Gj):
    signs=[-1,1]
    N_i= len(eig_Gi)
    N_j= len(eig_Gj)
    Mij = min(N_i, N_j)
    eigenvect_dist = np.zeros((Mij - 1))
    for r in range(1, Mij):
        temp=[]
        for  sign_s in signs:
            for sign_l in signs:
                v_ri = normalize_eigenv(sign_s * eig_Gi[r][1])
                v_rj = normalize_eigenv(sign_l * eig_Gj[r][1])
                temp.append(kruglov_distance(N_i,N_j,v_ri,v_rj))
        eigenvect_dist[r - 1] = min(temp)
    return(eigenvect_dist, np.sum(eigenvect_dist) / (Mij - 1)) 

def kruglov_distance(N_i,N_j,v_ri,v_rj):
    return (
        quad(
            integrand_kd, 0., 1., 
            epsabs = 1e-4, limit = 2000, 
            args=(N_i, N_j, v_ri, v_rj),
        )[0]
    )

def sgd_matrix(reign_lists):
    N_graphs = len(reign_lists)
    graph_dist_matx = np.zeros((N_graphs,len(reign_lists)))
    for i in range(N_graphs):
        for j in range(i, N_graphs):
            _, graph_dist = graphs_kruglov_distance(reign_lists[i],
                                                reign_lists[j])
            graph_dist_matx[i,j] = graph_dist
            graph_dist_matx[j,i] = graph_dist
    return graph_dist_matx

 

def hamming_distance(A_p0, A_p_list):
    n=len(A_p0)
    hamming_distance_list = np.zeros((len(A_p_list)))
    for i in range(len(A_p_list)):
        hamming_distance_list[i] = np.sum(np.abs(A_p0-A_p_list[i]))/(n*(n-1))
    return(hamming_distance_list)


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
    