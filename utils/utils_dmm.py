import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture
from scipy.stats import dirichlet
import math


def kmeans_init(data, k):
    n = len(data)
    p = len(data.columns)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)
    mu_not = kmeans.cluster_centers_
    #alsum=60
    alsum = p*5
    alpha_not = mu_not*alsum
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    return pi_not, alpha_not


def gmm_init(data, k):
    n = len(data)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(data)
    mu_not = clf.means_
    alsum = 60
    alpha_not = mu_not*alsum
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[clf.predict(data)[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    return pi_not, alpha_not


def random_init(data, k, random_seed=0):
    np.random.seed(random_seed)
    p = len(data.columns)
    alpha_not = []
    for h in range(k):
        alpha_not_temp = np.random.uniform(0, 50, p)
        alpha_not.append(alpha_not_temp)
    pi_not = sum(np.random.dirichlet([0.5 for i in range(k)], 1).tolist(), [])

    return pi_not, alpha_not


def kmeans_init_adv(data, k):
    n = len(data)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)
    #alpha_not=mu_not*alsum
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_new = []
    for i in range(k):
        data_cwise_ar = np.array(data_cwise[i])
        E = data_cwise_ar.mean(axis=0)
        E2 = (data_cwise_ar ** 2).mean(axis=0)
        E3 = ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E
        alpha_new.append(E3.tolist())
    return pi_not, alpha_new


def gmm_init_adv(data, k):
    n = len(data)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(data)
    #alpha_not=mu_not*alsum
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[clf.predict(data)[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_new = []
    for i in range(k):
        data_cwise_ar = np.array(data_cwise[i])
        E = data_cwise_ar.mean(axis=0)
        E2 = (data_cwise_ar ** 2).mean(axis=0)
        E3 = ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E
        alpha_new.append(E3.tolist())
    return pi_not, alpha_new


def dmm_loglikelihood(pi_temp, alpha_temp, data_lol):
    n = len(data_lol)
    k = len(alpha_temp)
    log_likelihood_values_temp = []
    for c in range(n):
        try:
            log_likelihood_old_temp = math.log(np.nansum(
                [pi_temp[f]*dirichlet.pdf(data_lol[c], alpha_temp[f]) for f in range(k)]))
        except:
            log_likelihood_old_temp = math.log1p(np.nansum(
                [pi_temp[f]*dirichlet.pdf(data_lol[c], alpha_temp[f]) for f in range(k)]))
        log_likelihood_values_temp.append(log_likelihood_old_temp)
    log_likelihood_old = np.sum(log_likelihood_values_temp)

    return log_likelihood_old


def dmm_responsibilities(pi_temp, alpha_temp, data_lol):
    n = len(data_lol)
    k = len(alpha_temp)
    gamma_temp = []
    for i in range(n):
        gamma_numer = []
        for j in range(k):
            temp_gamma_numer = (
                pi_temp[j]*dirichlet.pdf(data_lol[i], alpha_temp[j]))
            gamma_numer.append(temp_gamma_numer)
        gamma_row = gamma_numer / np.nansum(gamma_numer)
        gamma_temp.append(gamma_row)
    gamma_temp_ar = np.array(gamma_temp, dtype=np.float64)
    gamma_matrix = []
    for v in gamma_temp:
        gm_temp = v.tolist()
        gamma_matrix.append(gm_temp)
    return gamma_temp_ar, gamma_matrix


def dmm_pi_estimate(gamma_temp_ar):
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
