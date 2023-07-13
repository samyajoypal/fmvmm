import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import dirichlet as drm
from fmvmm.utils.utils_dmm import (kmeans_init, gmm_init, kmeans_init_adv, gmm_init_adv, random_init,
                             dmm_loglikelihood, dmm_responsibilities, dmm_pi_estimate)


from fmvmm.utils.utils_mixture import (mixture_clusters)


def estimate_alphas(data_cwise, alpha_not, method):
    alpha_new = []
    for t in range(len(data_cwise)):
        if np.array(data_cwise[t]).ndim == 1:
            alpha_new_temp = alpha_not[t]
        elif method == 'fixedpoint':
            alpha_new_temp = drm.mle(
                np.array(data_cwise[t]), method="fixedpoint", maxiter=9223372036854775807)
        else:
            alpha_new_temp = drm.mle(np.array(data_cwise[t]))

        alpha_new.append(alpha_new_temp)
    return alpha_new


class DMM:

    def __init__(self, number_of_clusters, sample, method="meanprecision", initialization="KMeans", tol=0.0001,show_loglikelihood_diff=False):
        self.number_of_clusters = number_of_clusters
        self.sample = sample
        self.method = method
        self.initialization = initialization
        self.tol = tol
        self.show_loglikelihood_diff=show_loglikelihood_diff
        self.k = self.number_of_clusters
        data = self.sample
        self.p = len(data.columns)
        self.n = len(data)

        self.data_lol = data.values.tolist()
        if self.initialization == "KMeans":
            self.pi_not, self.alpha_not = kmeans_init(data, self.k)
        elif self.initialization == "GMM":
            self.pi_not, self.alpha_not = gmm_init(data, self.k)
        elif self.initialization == "random":
            self.pi_not, self.alpha_not = random_init(data, self.k)

        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not

    def fit(self):
        log_like_diff = 5
        while log_like_diff > self.tol:

            log_likelihood_old = dmm_loglikelihood(
                self.pi_temp, self.alpha_temp, self.data_lol)
            gamma_temp_ar, gamma_matrix = dmm_responsibilities(
                self.pi_temp, self.alpha_temp, self.data_lol)
            pi_new = dmm_pi_estimate(gamma_temp_ar)

            cluster, data_cwise = mixture_clusters(gamma_matrix, self.data_lol)

            alpha_new = estimate_alphas(
                data_cwise, self.alpha_not, self.method)

            log_likelihood_new = dmm_loglikelihood(
                pi_new, alpha_new, self.data_lol)

            log_like_diff = abs(log_likelihood_new-log_likelihood_old)/log_likelihood_old
            estimated_mean = []
            for a in alpha_new:
                mean_temp = [b/np.sum(a) for b in a]
                estimated_mean.append(mean_temp)

            self.alpha_temp = alpha_new
            self.pi_temp = pi_new
            if self.show_loglikelihood_diff:
                print(log_like_diff)
        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.estimated_mean = estimated_mean
        self.cluster = cluster
        self.data_cwise = data_cwise
        self.gamma_temp_ar = gamma_temp_ar
        self.gamma_matrix = gamma_matrix
        self.log_likelihood_new = log_likelihood_new
        print("Hard DMM Fitting Done Successfully")

    def get_params(self):
        # print("The estimated pi values are ", self.pi_new)
        # print("The estimated alpha values are ", [l.tolist() for l in self.alpha_new])

        return self.pi_new, self.alpha_new

    def predict(self):
        return self.cluster

    def predict_new(self, x):
        data_lol = x.values.tolist()
        cluster, _ = mixture_clusters(self.gamma_matrix, data_lol)

        return cluster

    def estimated_mean(self):
        estimated_mean = pd.DataFrame(self.estimated_mean)
        return estimated_mean

    def responsibilities(self):

        return self.gamma_temp_ar

    def clustered_data(self):
        return pd.DataFrame(self.data_cwise).transpose()

    def bic(self):

        return ((self.k-1)+(self.k*self.p))*np.log(self.n) - 2*(self.log_likelihood_new)

    def aic(self):

        return 2*((self.k-1)+(self.k*self.p)) - 2*(self.log_likelihood_new)

    def icl(self):
        entropy_s=[]
        for i in self.gamma_matrix:
            for j in i:
                    ent_temp=j*np.log1p(j)
                    entropy_s.append(ent_temp)



        entropy=np.sum(entropy_s)

        return ((self.k-1)+(self.k*self.p))*np.log(self.n) - 2*(self.log_likelihood_new) - entropy
