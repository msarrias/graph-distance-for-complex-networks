import numpy as np
import random, time, math, json
from scipy.integrate import quad
import scipy.linalg as la
from collections import Counter
from matplotlib import pyplot as plt
import networkx as nx
from Graph import Graph

def fit_sgd_for_all_ref_models(first_k0, k0_list, sgd_obj, models_dict):
    sgd_dic = {}
    sgd_obj.eigenspectrum_G_ref()
    sgd_obj.eigenspectrum_G()
    print("")
    print("")
    print(f"SGD between G({first_k0}) and G(k)")
    print("")
    sgd_obj.fit_SGD()
    sgd_dic[first_k0] = sgd_obj.rep_sgd
    for k0 in k0_list[1:]:
        print("~~~~~~~~~~:~~~~~~~~~~")
        print(f"SGD between G({k0}) and G(k)")
        print("")
        sgd_obj.reset_G_ref(k0, models_dict['models_ref'][k0])
        sgd_obj.eigenspectrum_G_ref()
        sgd_obj.fit_SGD()
        sgd_dic[k0] = sgd_obj.rep_sgd
        print("")
        print("~~~~~~~~~~:~~~~~~~~~~")
    return(sgd_dic)

def fit_sgd_for_all_ref_modelsp(first_p0, p0_list, sgd_obj, models_dict):
    sgd_dic = {}
    sgd_obj.eigenspectrum_G_ref()
    sgd_obj.eigenspectrum_G()
    print("")
    print("")
    print(f"SGD between G({first_p0}) and G(p)")
    print("")
    sgd_obj.fit_SGD()
    sgd_dic[first_p0] = sgd_obj.rep_sgd
    for p0 in p0_list[1:]:
        print("~~~~~~~~~~:~~~~~~~~~~")
        print(f"SGD between G({p0}) and G(p)")
        print("")
        sgd_obj.reset_G_ref(p0, models_dict['models_ref'][p0])
        sgd_obj.eigenspectrum_G_ref()
        sgd_obj.fit_SGD()
        sgd_dic[p0] = sgd_obj.rep_sgd
        print("")
        print("~~~~~~~~~~:~~~~~~~~~~")
    return(sgd_dic)

#=========================SIMULATE-GRAPHS===============================#

def generate_dump_WS_models_file_fixed_p(n, p, replicates, k0_list,
                                         k_list, file_direct):
    #generate reference models
    WS_models_G0 = {}
    for k in k0_list:
        WS_models_G0[k] = nx.watts_strogatz_graph(n, int(k), p)
    #generate replicate models
    WS_models_Gk = {}
    for k in k_list:
        graph_dic = {}
        for j in range(replicates):
            graph_dic[j] = nx.watts_strogatz_graph(n, int(k), p)
        WS_models_Gk[k] = graph_dic
    #keep all in a dict
    WS_models_k = {}
    WS_models_k['models_ref'] = WS_models_G0
    WS_models_k['models'] = WS_models_Gk
    #dump in a json format file
    nx.write_gpickle(WS_models_k, file_direct)
    
def generate_dump_WS_models_file_fixed_k(n, k, replicates, p0_list,
                                         p_list, file_direct):
    #generate reference models
    WS_models_G0 = {}
    for p in p0_list:
        WS_models_G0[p] = nx.watts_strogatz_graph(n, k, p)
    #generate replicate models
    WS_models_Gp = {}
    for p in p_list:
        graph_dic = {}
        for j in range(replicates):
            graph_dic[j] = nx.watts_strogatz_graph(n, k, p)
        WS_models_Gp[p] = graph_dic
    #keep all in a dict
    WS_models_p = {}
    WS_models_p['models_ref'] = WS_models_G0
    WS_models_p['models'] = WS_models_Gp
    #dump in a json format file
    nx.write_gpickle(WS_models_p, file_direct)

def generate_dump_ER_models_fixed_n(p_list, p0_list, replicates,
                                    n, filedirect):
    #generate reference models
    ER_G0 = {}
    for p in p0_list:
        graph = ERB(n, p)
        while not nx.is_connected(graph):
            graph = ERB(n, p)
        ER_G0[p] = graph 
    #generate replicate models
    ER_G = {}
    for p in p_list:
        temp = {}
        for rep in range(replicates):
            graph = ERB(n, p)
            while not nx.is_connected(graph):
                graph = ERB(n, p)
            temp[rep] = graph
        ER_G[p] = temp
    ER_Graphs = {}
    ER_Graphs['models_ref'] = ER_G0
    ER_Graphs['models'] = ER_G
    nx.write_gpickle(ER_Graphs, filedirect)
    
#=========================A-MATRIX-OF-SIMULATED-GRAPHS=====================#
def compute_A_matrix(models_ref, models):
    A_dic_ref_models = {}
    for par_i in models_ref.keys():
        A_dic_ref_models[par_i] = (nx.adjacency_matrix(models_ref[par_i])).todense()
    A_models = {}
    for par_j, val_dic in models.items():
        A_dic_temp = {}
        for idx, it_graph in enumerate(list(val_dic.values())):
            A_dic_temp[idx] = (nx.adjacency_matrix(it_graph)).todense()
        A_models[par_j] = A_dic_temp
    A_dic = {}
    A_dic['models_ref'] = A_dic_ref_models
    A_dic['models'] = A_models
    return A_dic
#=========================SIMULATED-RESULTS===============================#
def avg_simulations_sgd_results(simulations_sgd_dic):
    avg_dic = {}
    std_dic = {}
    for key, value in simulations_sgd_dic.items():
        avg_dic[key] = [np.mean(value[i]) for i in value.keys()]
        std_dic[key] = [np.std(value[i]) for i in value.keys()]
    return avg_dic, std_dic

def compute_avg_hamming_distance(A_matrix):
    hd_model_0 = {}
    hd_model_avg_0 = {}
    par_list = list(A_matrix['models'].keys())
    replicates = len(A_matrix['models'][par_list[0]].keys())
    for ref, A_0 in A_matrix['models_ref'].items():
        hd_model_0[ref] = hamming_dist_models(A_0, A_matrix['models'], replicates)
        hd_model_avg_0[ref] = [np.mean(hd_model_0[ref][p]) for p in A_matrix['models'].keys()]
    return hd_model_avg_0

def mean_std_sim_dict(sim_dict):
    mean_dic = {}
    std_dic = {}
    for key, value in sim_dict.items():
        mean_dic[key] = [np.mean(value[i]) for i in value.keys()]
        std_dic[key] = [np.std(value[i]) for i in value.keys()]
    return mean_dic, std_dic
#=========================SIMULATED-MODELS-DEGREE-DISTRIBUTION============================#

def compute_dd_probd(A_dic):
    dd_k0_dic = {}
    prob_dd_k0_dic = {}
    lists = {}
    for key_, A in A_dic.items():
        dd_k0 = {}
        for key, value in A['models_ref'].items():
            dd_k0[key] = np.asarray(sum(value))[0]
        dd_k0_dic[key_] = dd_k0
        
        prob_dd_k0 = {}
        for key, value in dd_k0.items():
            temp_dd = Counter(value)
            prob_dd_k0[key] = {key: values / sum(temp_dd.values()) for key,
                               values in temp_dd.items()}
        prob_dd_k0_dic[key_] = prob_dd_k0
        
        lists[key_] = [sorted(zip(list(prob_dd_k0[k].keys()), 
                                  list(prob_dd_k0[k].values())), 
                              key=lambda x: x[0]) for k in list(prob_dd_k0.keys())]

    return dd_k0_dic, prob_dd_k0_dic, lists

def supports(sorted_list, k):
    return [sorted_list[k][i][0] for i in range(len(sorted_list[k]))]

def prob_distrib(sorted_list, k):
    if sum([sorted_list[k][i][1] 
            for i in range(len(sorted_list[k]))]) <= 1-1e-10:
        raise Exception('there is something wrong')
    else:
        return [sorted_list[k][i][1] for i in range(len(sorted_list[k]))]
    
#=================================================================================#
def ERB(N, p):
    g = nx.Graph() 
    g.add_nodes_from(range(1, N + 1)) 
    for i in g.nodes(): 
        for j in g.nodes(): 
            if (i < j):  
                eps = random.random() 
                if (eps < p): 
                    g.add_edge(i, j)
    return g

def erdos_renyi(N, p, weight = 1, seed = False): 
    if seed: random.seed(seed)       
    g = Graph(N)
    g.init_nodes()
    for i in g.nodes(): 
        for j in g.nodes(): 
            if (i < j):  
                eps = random.uniform(0,1)
                if (eps < p): 
                    g.add_edge(i, j, weight)
    return g
#=========================================================================#
def cdf_(N, y, v):
    tr = y - v
    return len(tr[tr>=0]) / N

def cdf(N,y,v):
    p = np.zeros((np.shape(y)))
    for i in range(len(y)):
        tr = y[i] - v
        p[i] =  len(tr[tr >= 0]) / N
    return(p)

def normalize_eigenv(eigenvc):
    return(
        (eigenvc - min(eigenvc)) / (max(eigenvc) - min(eigenvc))
    )

def integrand_kd(y, N_i, N_j, v_ri, v_rj):
    cdf_i = np.sum(y - v_ri >= 0) / N_i
    cdf_j = np.sum(y - v_rj >= 0) / N_j
    return np.abs(cdf_i - cdf_j)

def kruglov_distance(N_i, N_j, v_ri, v_rj):
    return (
        quad(
            integrand_kd, 0., 1., 
            epsabs = 1e-4, limit = 2000, 
            args=(N_i, N_j, v_ri, v_rj),
        )[0]
    )

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

    
def sort_eigenv(eigenvalues, eigenvectors):
    return sorted(zip(eigenvalues.real,
                      eigenvectors.T), key=lambda x: x[0])

def adj_degree_matrices(graph):
    A = (nx.adjacency_matrix(graph)).todense()
    D = np.diag(np.asarray(sum(A))[0])
    return A, D

def ctd_eigenv(vol_V, sorted_eigenv, D, eps):
    ctd_eign = {}
    for e in range(len(sorted_eigenv)):
        scale_factor = sorted_eigenv[e][0] * np.diagonal(D) + eps
        eigenv = np.sqrt( vol_V / scale_factor) * sorted_eigenv[e][1]
        ctd_eign[e] = [sorted_eigenv[e][0], eigenv]
    return ctd_eign

def eigen_decomp_WS_models(watts_strogatz_graphs_dic, replicates,
                           solve = "standard_L", ctd = False):
    eps=1e-10
    times_dic = {}
    eigenv_dic_p = {}
    A_dic = {}
    p = list(watts_strogatz_graphs_dic.keys())
    
    for ps in p:
        eigenv_dic = {}
        A_dic_p = {}
        tic_i = time.time() 
        times = []
        
        for j in range(replicates):
            graph = watts_strogatz_graphs_dic[ps][j]
            A_dic_p[j], D = adj_degree_matrices(graph)
            vol_V = sum(sum(D))
            
            if solve == "standard_L":
                eigenvalues, eigenvectors = la.eig(D - A_dic_p[j])
                eigenv_dic[j] = sort_eigenv(eigenvalues, eigenvectors) 
                
            if solve == "standard_Lsym":
                D_invsq = np.diag(1/np.sqrt(np.diag(D)))
                I = np.eye(D.shape[0])
                L_sym_rep = I - np.dot(D_invsq, A_dic_p[j]).dot(D_invsq)
                eigenvalues, eigenvectors = la.eig(L_sym_rep)
                
                if ctd:
                    eigenv_dic[j] = ctd_eigenv(vol_V, 
                                               sort_eigenv(eigenvalues,
                                                           eigenvectors),
                                               D, eps)
                else:
                    eigenv_dic[j] = sort_eigenv(eigenvalues,eigenvectors)
                
            if solve == "generalized":
                eigenvalues, eigenvectors = la.eig(D - A_dic_p[j], D)
                if ctd:
                    eigenv_dic[j] = ctd_eigenv(vol_V, 
                                               sort_eigenv(eigenvalues,
                                                           eigenvectors),
                                               D, eps)
                else:
                    eigenv_dic[j] = sort_eigenv(eigenvalues,eigenvectors)
                    
            tac_i = time.time()
            times.append((tac_i - tic_i) / 60)  
        A_dic[ps] =  A_dic_p
        eigenv_dic_p[ps] = eigenv_dic
        times_dic[ps] = times
    return (times_dic, A_dic, eigenv_dic_p)

def hamming_dist_models(A_p0, A_dic, replicates):
    haming_distance_dic = {}
    n=len(A_p0)
    p = list(A_dic.keys())
    for ps in p:
        A_ps = list(A_dic[ps].values())
        hamming_distances = []
        for j in range(replicates):
            adj_dif = np.abs(A_p0 - A_ps[j])
            hamming_distances.append(np.sum(adj_dif) / (n * (n - 1)))
        haming_distance_dic[ps] = hamming_distances
    return haming_distance_dic

def hamming_distance(A_p0, A_p_list):
    n=len(A_p0)
    hamming_distance_list = np.zeros((len(A_p_list)))
    for i in range(len(A_p_list)):
        adj_dif = np.abs(A_p0 - A_p_list[i])
        hamming_distance_list[i] = np.sum(adj_dif) / (n * (n - 1))
    return(hamming_distance_list)

                        
def spectral_distance(ctd_eignv_p0, ctd_eigenv_p_dic,
                      replicates, print_ = True):
                        
    G1 = ctd_eignv_p0
    Ni = len(G1)
    G2_dic = ctd_eigenv_p_dic
    p = list(G2_dic.keys())
    
    rep_distances_Lsym = {}
    integral_Lsym = []
    area_Lsym = []
    time_p_Lsym = []
    
    signs =[-1, 1]

    for idx, rp in enumerate(p):
        tic_p = time.time() 
        replicate_distance = []
        for rep in range(replicates):
            G2_dic_p  = G2_dic[rp][rep]
            Nj = len(G2_dic_p)
            Mij = min(Ni, Nj)
            integral_list = []
            for r in range(1, Mij):
                temp_integral = []
                for  sign_s in signs:
                    for sign_l in signs:
                        vri = sorted(normalize_eigenv(sign_s * G1[r][1]))
                        vrj = sorted(normalize_eigenv(sign_l * G2_dic_p[r][1]))
                        temp_integral.append(cdf_dist(vri, vrj))  
                integral_list.append(min(temp_integral))
            replicate_distance.append(sum(integral_list)/(Mij - 1))
        rep_distances_Lsym[rp] = replicate_distance
        tac_p = time.time()
        time_rp = (tac_p - tic_p) / 60
        if print_:
            print(f'd(G(0.1), G({rp})): {np.mean(rep_distances_Lsym[rp])}, ' \
                  f'took: {np.round(time_rp,4)} minutes')
        time_p_Lsym.append(time_rp)
        
    return (time_p_Lsym, rep_distances_Lsym)

def cdf_dist(vri, vrj):
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
        cdf_distance += (u[idx + 1] - u[idx]) * abs((vri_count / Ni) - (vrj_count / Nj))
        vri_count += vri_counter.get(sorted_elt, 0)
        vrj_count += vrj_counter.get(sorted_elt, 0)
    return cdf_distance

def pairwise_spectral_dist(eigen_graphi, eigen_graphj):
    signs =[-1, 1]
    Ni = len(eigen_graphi)
    Nj = len(eigen_graphj)
    Mij = min(Ni, Nj)
    integral_list = []
    integral_list_p = []
    for r in range(1, Mij):
        temp_integral = []
        temp_integral_py = []
        for  sign_s in signs:
            for sign_l in signs:
                vri = sorted(normalize_eigenv(sign_s * eigen_graphi[r][1]))
                vrj = sorted(normalize_eigenv(sign_l * eigen_graphj[r][1]))
                temp_integral.append(cdf_dist(vri, vrj))
                temp_integral_py.append(kruglov_distance(len(vri),len(vrj),
                                                         np.asarray(vri), np.asarray(vrj)))
        integral_list.append(min(temp_integral))
        integral_list_p.append(min(temp_integral_py))
    return(sum(integral_list_p)/(Mij - 1), sum(integral_list)/(Mij - 1))

def savage_plot(mean_list, p_list,  p0, std_list, hd_avg_list, ylims_sgd, y_lims_hd):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('p')
    ax1.set_ylabel('Spectral Graph Distance', color=color)
    ax1.plot(p_list, mean_list, '-o', color = color)
    ax1.axvline(x = p0, color = 'blue', linestyle = '--', label = "p_0 = "+ str(p0))
    ax1.fill_between(p_list, [mean_list[i] - std_list[i] for i in range(len(mean_list))],
                     [mean_list[i] + std_list[i] for i in range(len(mean_list))], 
                     color = 'lightgrey')
    ax1.set_ylim(ylims_sgd)
    ax1.tick_params(axis = 'y', labelcolor = color)

    ax2 = ax1.twinx()  

    color = 'tab:blue'
    ax2.set_ylabel('Hamming Distance ', color = color) 
    ax2.plot(p_list ,hd_avg_list, '-x', color = color)
    ax2.set_ylim(y_lims_hd)
    ax2.tick_params(axis = 'y', labelcolor = color)
    fig.tight_layout()
    plt.show()
# ===================================================================== #
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

def savage_plot_(graphs_list):
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
    
#=========================PLOT=RESULTS-SIMULATE-GRAPHS=====================#    
def plot_sgd_and_hammingd(ref_par_list, par_list, avg_dic,
                          std_dic, avg_hd, save_fig_list, par = 'p' ):
    fig = plt.figure(figsize=(20, 4))
    for i in range(len(ref_par_list)):
        par0 = ref_par_list[i]
        plt.subplot(1, 5, i+1)
        plt.tight_layout()
        if par == 'p':
            plt.title(r'$p_0$ = ' + str(par0))
            plt.xlabel('p')
        else:
            plt.title(r'$k_0$ = ' + str(par0))
            plt.xlabel('k')
        plt.ylabel('Spectral graph distance.')
        plt.plot(par_list, avg_dic[par0], '-o', color = 'red')
        plt.plot(par_list[np.argmin(avg_dic[par0])], 
                 np.min(avg_dic[par0]), 'x', color = 'blue')
        plt.axvline(x = par0, color = 'blue', linestyle = '--')
        minus_std = [avg_dic[par0][i] - std_dic[par0][i] 
                     for i in range(len(par_list))]
        plus_std = [avg_dic[par0][i] + std_dic[par0][i] 
                    for i in range(len(par_list))]
        plt.fill_between(par_list, minus_std, plus_std,
                         color = 'lightgrey')
    if save_fig_list[1] != 'none':
        plt.savefig(save_fig_list[0])
    fig = plt.figure(figsize=(20, 4))
    for i in range(len(ref_par_list)):
        par0 = ref_par_list[i]
        plt.subplot(1, 5, i+1)
        plt.tight_layout()
        if par == 'p': plt.xlabel('p')
        else: plt.xlabel('k')
        plt.ylabel('Hamming Distance')
        plt.axvline(x = par0, color = 'blue', linestyle = '--')
        plt.plot(par_list, avg_hd[par0], '-o', color = 'red')
    if save_fig_list[1] != 'none':
        plt.savefig(save_fig_list[1])
