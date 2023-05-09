import numpy as np
import pandas as pd
from utils.utils_fmm import (fmm_kmeans_init, fmm_loglikelihood, fmm_responsibilities,
                             fmm_pi_estimate, fmm_estimate_alphas, fmm_aic, fmm_bic)


from utils.utils_mixture import (mixture_clusters)
import math

import itertools

from mvem.stats import multivariate_skewnorm as mvsn
from mvem.stats import multivariate_genhyperbolic as mghp
from mvem.stats import multivariate_genskewt as mgst
from mvem.stats import multivariate_hyperbolic as mvhb
from mvem.stats import multivariate_norm as mvn
from mvem.stats import multivariate_norminvgauss as mnig
from mvem.stats import multivariate_t as mvt
from mvem.stats import multivariate_vargamma as mvvg
import itertools

dist_map = {"mvsn": mvsn, "mghp": mghp, "mgst": mgst, "mvhb": mvhb,
            "mvn": mvn, "mnig": mnig, "mvt": mvt, "mvvg": mvvg}

all_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg"]


def add_elements_at_indices(my_list, indices_to_add, new_elements):
    for i, index in enumerate(indices_to_add):
        my_list.insert(index+i, new_elements[i])
    return my_list

def flatten(seq):
    flat_list = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):  # Check if item is a list, tuple or ndarray
            flat_list.extend(flatten(item))  # Recursively call flatten on the nested list, tuple or ndarray
        else:
            flat_list.append(item)  # Append non-list/tuple/ndarray items to the flat list
    return flat_list


def remove_nan_tuples(list_of_lists, list_of_list_of_tuples):
    # Initialize a list to store indices of sub-lists or tuples to remove
    indices_to_remove = []

    for i in range(len(list_of_list_of_tuples)):
        temp=flatten(list_of_list_of_tuples[i])
        #temp=flatten(temp)
        # for j in range(len(list_of_list_of_tuples[i])):
        #     for k in range(len(list_of_list_of_tuples[i][j])):
        #         for l in range(len(list_of_list_of_tuples[i][j][k])):
        #             temp.append(list_of_list_of_tuples[i][j][k][l])
        # temp=list(itertools.chain(*temp))
        # temp = np.array(temp, dtype=float)
        if any(np.isnan(temp)):
            indices_to_remove.append(i)
            

    # Remove sub-lists and tuples using the stored indices
    if len(indices_to_remove)>0:
        list_of_lists = [list_of_lists[i] for i in range(len(list_of_lists)) if i not in indices_to_remove]
        list_of_list_of_tuples = [list_of_list_of_tuples[i] for i in range(len(list_of_list_of_tuples)) if i not in indices_to_remove]
        
        # for index in indices_to_remove:
        #     del list_of_lists[index]
        #     del list_of_list_of_tuples[index]

    return list_of_lists, list_of_list_of_tuples, indices_to_remove

def remove_elements_by_index(my_list, index_list):
    if len(index_list)>0:
        return [item for i, item in enumerate(my_list) if i not in index_list]
    else:
        return my_list

def get_dist_names(dist_comb):
    return [str(c.__name__) for c in dist_comb]

class fmvmm():
    def __init__(self, number_of_clusters, sample, tol=0.00001, list_of_dist=all_dist,specific_comb=False,print_log_likelihood=False):
        self.k = number_of_clusters
        self.sample = sample
        self.list_of_dist = list_of_dist
        self.data = sample
        self.p = len(sample.columns)
        self.n = len(sample)
        self.dist_variables = [dist_map[list_of_dist[j]]
                               for j in range(len(list_of_dist))]
        self.specific_comb=specific_comb
        if self.specific_comb==True:
            self.dist_combs = list(
                itertools.combinations(self.dist_variables, self.k))
        else:
            self.dist_combs = list(
                itertools.combinations_with_replacement(self.dist_variables, self.k))
        self.data_lol = self.data.values.tolist()
        self.tol=tol
        self.print_log_likelihood=print_log_likelihood
        #print(self.dist_combs)

    def fit(self):
        self.list_aic = []
        self.list_bic = []
        self.list_pi = []
        self.list_alpha = []
        self.list_cluster = []
        self.list_log_likelihood = []
        self.list_gamma_matrix = []
        self.not_worked_dist=[]
        self.worked_dist=[]
        for l in range(len(self.dist_combs)):
            try:
                
                self.pi_not, self.alpha_not = fmm_kmeans_init(
                    self.data, self.k, self.dist_combs[l])
                self.alpha_temp = self.alpha_not
                self.pi_temp = self.pi_not
                # print(self.pi_temp[0])
                # print(self.alpha_temp[0])
                # print(self.dist_combs[0][0])
                # print(math.log(np.nansum(
                #     [self.pi_temp[f]*self.dist_combs[0][f].pdf(np.reshape(np.array(self.data_lol[0]),(1,len(self.data_lol[0]))), *self.alpha_temp[f]) for f in range(self.k)])))
    
                log_like_diff = 5
    
                while log_like_diff > self.tol:
                    log_likelihood_old = fmm_loglikelihood(
                        self.pi_temp, self.alpha_temp, self.data_lol, self.dist_combs[l])
                    gamma_temp_ar, gamma_matrix = fmm_responsibilities(
                        self.pi_temp, self.alpha_temp, self.data_lol, self.dist_combs[l])
                    pi_new = fmm_pi_estimate(gamma_temp_ar)
    
                    cluster, data_cwise = mixture_clusters(
                        gamma_matrix, self.data_lol)
    
                    alpha_new = fmm_estimate_alphas(
                        data_cwise, self.alpha_not, self.dist_combs[l])
    
                    log_likelihood_new = fmm_loglikelihood(
                        pi_new, alpha_new, self.data_lol, self.dist_combs[l])
    
                    log_like_diff = abs(log_likelihood_new-log_likelihood_old)
                    self.alpha_temp = alpha_new
                    self.pi_temp = pi_new
                    if self.print_log_likelihood==True:
                        print("Loglikelihood:",log_likelihood_new, "Difference:", log_like_diff )
    
                self.list_aic.append(
                    fmm_aic(self.k, alpha_new, log_likelihood_new))
                self.list_bic.append(
                    fmm_bic(self.k, alpha_new, log_likelihood_new, self.n))
                self.list_pi.append(pi_new)
                self.list_alpha.append(alpha_new)
                self.list_cluster.append(cluster)
                self.list_log_likelihood.append(log_likelihood_new)
                self.list_gamma_matrix.append(gamma_matrix)
                self.worked_dist.append(self.dist_combs[l])
                print("distribution fitted", get_dist_names(self.dist_combs[l]))
            except:
                self.not_worked_dist.append(self.dist_combs[l])
                pass
            
        self.list_pi,self.list_alpha,self.nan_ind=remove_nan_tuples(self.list_pi,self.list_alpha)
        self.list_aic=remove_elements_by_index(self.list_aic,self.nan_ind)
        self.list_bic=remove_elements_by_index(self.list_bic, self.nan_ind)
        self.list_cluster=remove_elements_by_index(self.list_cluster, self.nan_ind)
        self.list_log_likelihood=remove_elements_by_index(self.list_log_likelihood, self.nan_ind)
        self.list_gamma_matrix=remove_elements_by_index(self.list_gamma_matrix, self.nan_ind)
        self.not_worked_dist.extend(w for w in [self.worked_dist[h] for h in self.nan_ind])
        self.worked_dist=remove_elements_by_index(self.worked_dist, self.nan_ind)
        print("Model fitted successfully")

    def get_params(self):
        return self.list_pi, self.list_alpha

    def predict(self):
        return self.list_cluster

    def predict_new(self, x):
        data_lol = x.values.tolist()
        cluster_all = []
        for l in range(len(self.dist_combs)):
            cluster, _ = mixture_clusters(self.gamma_matrix[l], data_lol)
            cluster_all.append(cluster)

        return cluster_all

    def bic(self):
        return self.list_bic

    def aic(self):
        return self.list_aic

    def best_mixture(self):
        best_mix = self.worked_dist[np.argmin(self.list_aic)]
        return [str(best_mix[i].__name__) for i in range(len(best_mix))]

    def best_params(self):
        return self.list_pi[np.argmin(self.list_aic)], self.list_alpha[np.argmin(self.list_aic)]

    def best_predict(self):
        return self.list_cluster[np.argmin(self.list_aic)]

    def best_predict_new(self, x):
        data_lol = x.values.tolist()
        cluster_all = []
        for l in range(len(self.dist_combs)):
            cluster, _ = mixture_clusters(self.gamma_matrix[l], data_lol)
            cluster_all.append(cluster)
        return cluster_all[np.argmin(self.list_aic)]

    def best_aic(self):
        return np.min(self.list_aic)

    def best_bic(self):
        return np.min(self.list_bic)
    def not_worked(self):
        print("Distribution Combinations That Could Not Be Fitted:")
        for i in range(len(self.not_worked_dist)):
            print(i, [str(self.not_worked_dist[i][j].__name__) for j in range(len(self.not_worked_dist[i]))])
    
    def worked(self):
        print("Distribution Combinations That Could Be Fitted:")
        for i in range(len(self.worked_dist)):
            print(i, [str(self.worked_dist[i][j].__name__) for j in range(len(self.worked_dist[i]))])
            
            
            
            
            
            