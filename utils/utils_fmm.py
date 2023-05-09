import numpy as np
import pandas as pd
import itertools
import math
from sklearn.cluster import KMeans

def fmm_loglikelihood(pi_temp,alpha_temp,data_lol,dist_comb):
    n = len(data_lol)
    k = len(alpha_temp)
    log_likelihood_values_temp = []
    for c in range(n):
        try:
            log_likelihood_old_temp = math.log(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        except:
            log_likelihood_old_temp = math.log1p(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        log_likelihood_values_temp.append(log_likelihood_old_temp)
    log_likelihood_old = np.sum(log_likelihood_values_temp)

    return log_likelihood_old


def fmm_responsibilities(pi_temp, alpha_temp, data_lol,dist_comb):
    n = len(data_lol)
    k = len(alpha_temp)
    gamma_temp = []
    for i in range(n):
        gamma_numer = []
        for j in range(k):
            temp_gamma_numer = (
                pi_temp[j]*dist_comb[j].pdf(np.reshape(np.array(data_lol[i]),(1,len(data_lol[i]))), *alpha_temp[j]))
            gamma_numer.append(temp_gamma_numer)
        gamma_row = gamma_numer / np.nansum(np.asanyarray(gamma_numer))
        gamma_temp.append(gamma_row)
    gamma_temp_ar = np.array(gamma_temp, dtype=np.float64)
    gamma_matrix = []
    for v in gamma_temp:
        gm_temp = v.tolist()
        gamma_matrix.append(gm_temp)
    return gamma_temp_ar, gamma_matrix


def fmm_pi_estimate(gamma_temp_ar):
    n = gamma_temp_ar.shape[0]
    k = gamma_temp_ar.shape[1]
    pi_new = []
    nk = []
    for g in range(k):
        nk_temp = np.nansum([gamma_temp_ar[w, g] for w in range(n)])
        pi_temp = nk_temp/n
        pi_new.append(pi_temp)
        nk.append(nk_temp)
    return pi_new


def fmm_kmeans_init(data, k,dist_comb):
    n = len(data)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_not=[dist_comb[j].fit(np.array(data_cwise[j])) for j in range(k)]
    return pi_not, alpha_not


def fmm_estimate_alphas(data_cwise, alpha_not, dist_comb):
    alpha_new = []
    for t in range(len(data_cwise)):
        if np.array(data_cwise[t]).ndim == 1:
            alpha_new_temp = alpha_not[t]
        else:
            alpha_new_temp = dist_comb[t].fit(np.array(np.array(data_cwise[t])))

        alpha_new.append(alpha_new_temp)
    return alpha_new

def fmm_aic(k,alpha_new,log_likelihood_new):
    return 2*((k-1)+np.sum([len(alpha_new[j]) for j in range(k)])) - 2*log_likelihood_new

def fmm_bic(k,alpha_new,log_likelihood_new,n):
    return ((k-1)+np.sum([len(alpha_new[j]) for j in range(k)]))*np.log(n) - 2*log_likelihood_new

