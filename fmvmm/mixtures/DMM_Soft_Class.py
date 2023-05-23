
import sys

import numpy as np
from numpy import (
    asanyarray,
    exp
)
from numpy.linalg import norm
from scipy.special import polygamma, psi
from scipy.special import digamma
import pandas as pd
import copy
from fmvmm.utils.utils_dmm import (kmeans_init, gmm_init, kmeans_init_adv, gmm_init_adv, random_init,
                             dmm_loglikelihood, dmm_responsibilities, dmm_pi_estimate)

from fmvmm.utils.utils_mixture import (mixture_clusters)
MAXINT = sys.maxsize
euler = -1 * psi(1)  # Euler-Mascheroni constant


class NotConvergingError(Exception):
    """Error when a successive approximation method doesn't converge
    """
    pass


def _ipsi(y, tol=1.48e-9, maxiter=10):
    """Inverse of psi (digamma) using Newton's method. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).

    Parameters
    ----------
    y : (K,) shape array
        y-values of psi(x)
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 10.

    Returns
    -------
    (K,) shape array
        Approximate x for psi(x)."""
    y = asanyarray(y, dtype="float")
    x0 = np.piecewise(
        y,
        [y >= -2.22, y < -2.22],
        [(lambda x: exp(x) + 0.5), (lambda x: -1 / (x + euler))],
    )
    for i in range(maxiter):
        x1 = x0 - (psi(x0) - y) / _trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, " f"value is {x1}")


def _trigamma(x):
    return polygamma(1, x)


def dirichlet_mix_mle(x: np.ndarray, gamma: np.ndarray, alpha_init):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    alpha_not = alpha_init
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    alpha_old = alpha_not
    alpha_all = []
    for o in range(k):

        log_x = []
        for f in range(p):
            log_x_temp = (
                1/N_js[o])*np.sum([np.log(x_lol[j][f])*gamma[j, o] for j in range(n)])
            log_x.append(log_x_temp)
        alpha_new = []
        for i in range(p):
            diff = 5
            t = copy.deepcopy(alpha_old[o][i])
            t1 = copy.deepcopy(alpha_old[o][i])
            di_alpha_not = digamma(t)
            di_alpha_old = di_alpha_not
            beta = np.sum(alpha_old[o])-t
            if beta > 500:
                alpha_new_temp = _ipsi(
                    np.log(beta)+log_x[i], tol=1.48e-9, maxiter=10000)
            else:

                while diff > 0.0001:
                    di_alpha_sum = digamma(np.sum(alpha_old[o])-t + t1)
                    di_alpha_new = di_alpha_sum + log_x[i]
                    t1 = _ipsi(di_alpha_new, tol=1.48e-9, maxiter=1000)
                    diff = abs(di_alpha_new - di_alpha_old)
                    di_alpha_old = di_alpha_new
                    alpha_new_temp = t1
            alpha_new.append(alpha_new_temp)
        alpha_all.append(alpha_new)

    return alpha_all


def dirichlet_mean_precision_mle(x: np.ndarray, gamma: np.ndarray, alpha_init):
    alpha_old = alpha_init
    diff = 5
    norm0 = 5
    while diff > 0.0001:
        alpha_1 = _fit_s(x, gamma, alpha_old)
        alpha_2 = _fit_m(x, gamma, alpha_1)
        norm1 = norm(np.array(alpha_2)-np.array(alpha_old))
        diff = abs(norm1-norm0)
        #print("alpha_diff",diff)
        norm0 = norm1
        alpha_old = alpha_2
    return alpha_2


def _fit_s(x: np.ndarray, gamma: np.ndarray, alpha_old):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
              for i in range(len(alpha_old))]
    alpha_new_all = []
    for o in range(k):
        diff = 5
        sjold = sj_old[o]
        while diff > 0.0001:
            cs = []
            for i in range(n):
                for j in range(p):
                    c_temp = mj_old[o][j]*gamma[i, o]*np.log(x_lol[i][j])
                    cs.append(c_temp)
            c = np.sum(cs)
            bs = []
            for i in range(p):
                b_temp = mj_old[o][i]*digamma(sjold*mj_old[o][i])
                bs.append(b_temp)
            b = N_js[o]*np.sum(bs)
            a = N_js[o]*digamma(sjold)
            firststdif = a-b+c
            d = N_js[o]*_trigamma(sjold)
            es = []
            for i in range(p):
                es_temp = (mj_old[o][i]**2)*_trigamma(sjold*mj_old[o][i])
                es.append(es_temp)
            e = N_js[o]*np.sum(es)
            seconddif = d-e
            if seconddif < 0:
                sj_new = sjold - firststdif/seconddif
            elif firststdif + sjold*seconddif < 0:
                sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
            else:
                raise NotConvergingError(f"Unable to update s from {sjold}")
            diff = abs(sj_new-sjold)
            #print("s_diff",diff)
            sjold = sj_new
        alphaj_new = [sj_new*mj_old[o][i] for i in range(p)]
        alpha_new_all.append(alphaj_new)
    return alpha_new_all


def _fit_m(x: np.ndarray, gamma: np.ndarray, alpha_old):
    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
              for i in range(len(alpha_old))]
    alpha_new_all = []

    for o in range(k):

        sjold = sj_old[o]
        mjold = mj_old[o]
        alpha_j_old = alpha_old[o]
        m6 = []
        for s in range(p):
            m5 = np.sum([gamma[i, o]*np.log(x_lol[i][s])
                        for i in range(n)])/N_js[o]
            m6.append(m5)
        diff = 5
        norm0 = 5
        while diff > 0.0001:
            alpha_j_new = []
            for j in range(p):
                m1 = m6[j]
                m2 = [mjold[q]*m6[q]for q in range(p)]
                m3 = np.sum(m2)
                m4 = np.sum([mjold[r]*digamma(sjold*mjold[r])
                            for r in range(p)])
                alpha_jm_new = _ipsi(m1-m3+m4, tol=1.48e-9, maxiter=1000)
                alpha_j_new.append(alpha_jm_new)
            alpha_j_new = np.array(alpha_j_new) / \
                np.array(alpha_j_new).sum() * sjold
            norm1 = norm(alpha_j_new-alpha_j_old)
            diff = abs(norm1-norm0)
            #print("m_diff",diff)
            norm0 = norm1
            mjold = alpha_j_new/alpha_j_new.sum()
        alpha_new_all.append(alpha_j_new.tolist())
    return alpha_new_all


def estimate_alphas(data: np.ndarray, gamma_temp_ar, alpha_temp, method):
    if method == "fixediteration":
        alpha_new = dirichlet_mix_mle(data, gamma_temp_ar, alpha_temp)
    else:
        alpha_new = dirichlet_mean_precision_mle(
            data, gamma_temp_ar, alpha_temp)
    return alpha_new


class DMM_Soft:

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
        np.random.seed(0)
        if self.initialization == "KMeans":
            self.pi_not, self.alpha_not = kmeans_init(data, self.k)
        elif self.initialization == "GMM":
            self.pi_not, self.alpha_not = gmm_init(data, self.k)
        elif self.initialization == "KMeans Adv":
            self.pi_not, self.alpha_not = kmeans_init_adv(data, self.k)
        elif self.initialization == "GMM Adv":
            self.pi_not, self.alpha_not = gmm_init_adv(data, self.k)
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
                self.sample.to_numpy(), gamma_temp_ar, self.alpha_temp, self.method)

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
        print("Soft DMM Fitting Done Successfully")

    def get_params(self):
        #print("The estimated pi values are ", self.pi_new)
        #print("The estimated alpha values are ", self.alpha_new)

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
