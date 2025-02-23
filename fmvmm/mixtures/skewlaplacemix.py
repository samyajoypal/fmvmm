"""
some matlab codes were received personally from Prof.Fatma Zehra Doğru,
author of Finite Mixtures of Multivariate Skew Laplace Distributions.
Many thanks to her. Rest was incorporated from the paper itself.

"""

import numpy as np
from fmvmm.mixtures._base import BaseMixture
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time
from fmvmm.utils.utils_mixture import (mixture_clusters)
from numpy.linalg import cholesky
import math
from scipy.special import gamma as gamma_func

###############################################################################
# (A) Generate random samples from a SINGLE Multivariate Skew-Laplace
###############################################################################

def sample_skewlaplace(mu, Sigma, gamma,n, random_state=None):
    """
    Generate 'n' samples in R^p from a single p-variate Skew-Laplace distribution.

    The model:
      1) v_i ~ Gamma( (p+1)/2, scale=2 )
      2) z_i ~ Normal(0, I_p)
      3) x_i = mu + (1 / v_i)*gamma + sqrt(1 / v_i)* L * z_i
         where L = chol(Sigma).

    Parameters
    ----------
    n : int
        Number of samples
    mu : (p,) array
        Location vector
    Sigma : (p,p) array
        Positive-definite covariance
    gamma : (p,) array
        Skewness vector
    random_state : int or None
        If set, seed for reproducibility.

    Returns
    -------
    X : array, shape (n, p)
    """
    # Convert mu, gamma to NumPy arrays (if not already)
    mu = np.asarray(mu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    
    rng = np.random.default_rng(random_state)
    p = len(mu)

    # 1) v_i = 1 / Gamma( (p+1)/2, scale=2 )
    # Actually we want v_i = 1 / g_i, where g_i ~ Gamma((p+1)/2, scale=2).
    # => v_i = 1 / g_i
    g_samples = rng.gamma(shape=(p+1)/2.0, scale=2.0, size=n)
    v_samples = 1.0 / g_samples

    # 2) z_i ~ Normal(0, I_p)
    z_samples = rng.normal(size=(n, p))

    # 3) L = chol(Sigma)
    L = cholesky(Sigma)

    # 4) x_i = mu + (1/v_i)*gamma + sqrt(1/v_i)* L * z_i
    X = np.zeros((n, p))
    for i in range(n):
        X[i] = (
            mu
            + v_samples[i] * gamma
            + np.sqrt(v_samples[i]) * (L @ z_samples[i])
        )

    return X


def compute_empirical_info_and_se(model, pi = None, alpha = None, data = None, gamma_res = None, use_model = True):
    """
    Given a fitted SkewLaplaceMix model, compute:
      1) Empirical information matrix I_e = sum_j s_j s_j^T,
      2) Covariance = (I_e)^{-1},
      3) Standard errors = sqrt(diag(Cov)).

    Returns
    -------
    SE : (d,) array
        Standard errors for each parameter in the vector of parameters
        (in the same order we construct the score).
    Cov : (d, d) array
        Approx. covariance matrix = I_e^{-1}.
    """
    # 1) Extract final parameters from the model
    #    pi_k: shape (K,)
    #    alpha_est: list of dicts or list-lists: alpha_est[k] = [mu_k, Sigma_k, gamma_k]
    if use_model:
        pi_est = model.pi_new
    else:
        pi_est = pi
    if use_model:
        alpha_est = model.alpha_new  # length K
    else:
        alpha_est = alpha
        
    

    K = len(pi_est)
    # alpha_est[k][0] => mu_k
    # alpha_est[k][1] => Sigma_k
    # alpha_est[k][2] => gamma_k
    p = alpha_est[0][0].shape[0]

    # 2) Data & final responsibilities
    if use_model:
        X = model.data
    else:
        X = data
    n = X.shape[0]
    # responsibilities => shape (n, K)
    if use_model:
        Z = model.gamma_temp_ar  # e.g. exp(log_gamma_new)
    else:
        Z = gamma_res

    # 3) Compute v1_{i,k}, v2_{i,k}
    V1 = np.zeros((n, K))
    V2 = np.zeros((n, K))

    inv_Sigmas = []
    a_values = []
    for k_i in range(K):
        Sigma_k = alpha_est[k_i][1]
        inv_Sig = np.linalg.inv(Sigma_k)
        inv_Sigmas.append(inv_Sig)

        g_k = alpha_est[k_i][2]
        a_k = np.sqrt(1.0 + g_k @ inv_Sig @ g_k)
        a_values.append(a_k)

    for i in range(n):
        x_i = X[i]
        for k_i in range(K):
            mu_k = alpha_est[k_i][0]
            g_k  = alpha_est[k_i][2]
            invS_k = inv_Sigmas[k_i]
            a_k = a_values[k_i]

            diff = x_i - mu_k
            dist_sq = diff @ invS_k @ diff
            dist = np.sqrt(max(dist_sq, 1e-14))

            # v1_{i,k} = a_k / dist
            # v2_{i,k} = [1 + sqrt(a_k^2 * dist^2)] / a_k^2
            V1[i, k_i] = a_k / dist
            V2[i, k_i] = (1.0 + np.sqrt(a_k*a_k * dist_sq)) / (a_k*a_k)

    # 4) Score function for a single observation
    def score_single_observation(x, z_row, v1_row, v2_row):
        """
        Returns the score vector s_j for one observation j,
        given z_{j,k}, v1_{j,k}, v2_{j,k}, pi_k, alpha[k].
        Order:
          - (K-1) partials wrt pi_0.. pi_{K-2},
          - For each k: partial wrt mu_k, Sigma_k, gamma_k
        """
        score_list = []

        # (A) partial wrt pi_k => eqn(4.2)
        # Each is a scalar => wrap in np.array([ ... ])
        for r in range(K-1):
            val = z_row[r]/pi_est[r] - z_row[K-1]/pi_est[K-1]
            score_list.append(np.array([val]))  # ensure shape (1,)

        # (B) partial wrt mu_k, Sigma_k, gamma_k
        for k_i in range(K):
            mu_k = alpha_est[k_i][0]
            Sigma_k = alpha_est[k_i][1]
            g_k = alpha_est[k_i][2]
            invS_k = inv_Sigmas[k_i]

            diff = x - mu_k

            # eqn(4.3) => partial wrt mu_k => shape (p,)
            s_mu_k = (
                z_row[k_i]*v1_row[k_i]*(invS_k @ diff)
                - z_row[k_i]*v2_row[k_i]*(invS_k @ g_k)
            )

            # eqn(4.4) => partial wrt Sigma_k => shape (p(p+1)/2,)
            GkGkT = np.outer(g_k, g_k)
            outer_diff = np.outer(diff, diff)
            
            A = (
                invS_k
                - v1_row[k_i] * (invS_k @ outer_diff @ invS_k)
                - v2_row[k_i] * (invS_k @ GkGkT   @ invS_k)
            )
            
            
            
            A_diag = np.diag(np.diag(A))          # shape (p,p) with only diagonal of A
            A_corrected = -A + 0.5 * A_diag       # the bracket in eqn. (4.4)
            sSigma_full = z_row[k_i] * A_corrected
            
            # Flatten with vech-lower
            sSigma_k_list = []
            for ii in range(p):
                for jj in range(ii, p):
                    sSigma_k_list.append(sSigma_full[ii, jj])
            sSigma_k = np.array(sSigma_k_list)

            # eqn(4.5) => partial wrt gamma_k => shape (p,)
            s_gamma_k = (
                z_row[k_i]*v1_row[k_i]*(invS_k @ diff)
                - z_row[k_i]*v2_row[k_i]*(invS_k @ g_k)
            )

            score_list.append(s_mu_k)
            score_list.append(sSigma_k)
            score_list.append(s_gamma_k)

        # Now everything in score_list is a 1D array => we can safely concatenate
        return np.concatenate(score_list)

    # 5) Sum up s_j s_j^T over j=1..n
    test_s = score_single_observation(X[0], Z[0], V1[0], V2[0])
    d = len(test_s)

    I_e = np.zeros((d, d))
    for j in range(n):
        s_j = score_single_observation(X[j], Z[j], V1[j], V2[j])
        I_e += np.outer(s_j, s_j)

    # 6) Cov = I_e^{-1}, SE = sqrt(diag(Cov))
    # eigenvalues = np.linalg.eigvalsh(I_e)
    # if np.any(eigenvalues <= 0):
    #     _reg = 1e-6  # Small regularization term
    #     I_e = I_e + _reg * np.eye(I_e.shape[0])
    Cov = np.linalg.pinv(I_e)
    SE = np.sqrt(np.diag(Cov))

    return I_e, SE




def k_means_init_skewlaplace(X, k):
    """
    1) Run KMeans to get cluster assignments.
    2) Compute each cluster's mean, covariance.
    3) gamma_k = 0 (shape vector).
    4) pi_k = #points in cluster / n.
    """
    n, p = X.shape

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_  # shape (k, p)

    # pi array
    pi_init = np.zeros(k)
    alpha_init = []
    for cluster_id in range(k):
        # collect points in that cluster
        idx = np.where(labels == cluster_id)[0]
        pi_init[cluster_id] = len(idx) / n

        # mean = cluster center
        mu_k = centers[cluster_id]

        # covariance => empirical from data in cluster
        if len(idx) > 1:
            cov_k = np.cov(X[idx].T, bias=False)  # shape (p,p)
        else:
            # if only one point, fallback to identity or very small
            cov_k = np.eye(p)

        # gamma_k => np.zeros(p)
        gamma_k = np.zeros(p)

        alpha_init.append((mu_k, cov_k, gamma_k))

    return pi_init, alpha_init

def gmm_init_skewlaplace(X, k):
    """
    1) Run GaussianMixture to get cluster assignments.
    2) Extract means, covariances, weights.
    3) gamma_k = 0.
    4) pi_k from GMM.
    """
    n, p = X.shape

    gm = GaussianMixture(n_components=k, random_state=0).fit(X)
    pi_init = gm.weights_  # shape (k,)
    centers = gm.means_    # shape (k, p)
    covariances = gm.covariances_  # shape (k, p, p)

    alpha_init = []
    for cluster_id in range(k):
        mu_k = centers[cluster_id]
        cov_k = covariances[cluster_id]
        gamma_k = np.zeros(p)
        alpha_init.append(( mu_k,cov_k, gamma_k))

    return pi_init, alpha_init

def estimate_alphas_skewlaplace(X, gamma_matrix, alpha_old):
    """
    M-step update for a K-component Multivariate Skew-Laplace mixture,
    using the same formulas as your 2-component MATLAB code but generalized.

    Parameters
    ----------
    X : array, shape (n, p)
        Data matrix.
    gamma_matrix : array, shape (n, K)
        Responsibilities (E-step), gamma_matrix[i, k] = posterior probability that
        sample i belongs to cluster k.
    alpha_old : list of dict, length K
        alpha_old[k] = {"mu": (p,), "Sigma": (p,p), "gamma": (p,)} - old parameters.

    Returns
    -------
    alpha_new : list of dict, length K
        Updated parameter set with structure matching alpha_old.
    """

    n, p = X.shape
    K = len(alpha_old)

    # We will store new parameters in alpha_new
    alpha_new = [None]*K

    # For convenience, precompute inverses and a_k = sqrt(1 + gamma_k^T S_k^-1 gamma_k)
    inv_Sigma_old = []
    a_array = np.zeros(K)
    for k in range(K):
        Sk_inv = np.linalg.inv(alpha_old[k][1])
        inv_Sigma_old.append(Sk_inv)
        gk_old = alpha_old[k][2]
        a_array[k] = np.sqrt(1.0 + gk_old @ Sk_inv @ gk_old)

    # We will accumulate partial sums for each cluster k.
    # For clarity, define arrays of shape (K,) or (K,p) or (K,p,p) as needed.
    sum_zk          = np.zeros(K)       # sum_i gamma_{i,k}
    sum_zk_v1       = np.zeros(K)       # sum_i gamma_{i,k} * v1_{i,k}
    sum_zk_v2       = np.zeros(K)       # sum_i gamma_{i,k} * v2_{i,k}
    sum_zk_x        = np.zeros((K, p))   # sum_i [gamma_{i,k} * x_i]
    sum_zk_v1_x     = np.zeros((K, p))   # sum_i [gamma_{i,k} * v1_{i,k} * x_i]
    # For Sigma:
    # We'll accumulate the numerators S_k_num and the denominators S_k_den:
    S_k_num = [np.zeros((p, p)) for _ in range(K)]
    S_k_den = np.zeros(K)

    # Loop over data points once, computing v1_{i,k}, v2_{i,k} and partial sums.
    for i in range(n):
        xi = X[i]

        for k in range(K):
            gk_old = alpha_old[k][2]   # shape (p,)
            muk_old = alpha_old[k][0]     # shape (p,)
            Sk_inv = inv_Sigma_old[k]
            ak = a_array[k]

            # Distance using old mu, old Sigma:
            diff = xi - muk_old
            dist_sq = diff @ Sk_inv @ diff
            dist_val = np.sqrt(max(dist_sq, 1e-14))  # clip to avoid /0

            # v1_{i,k} and v2_{i,k}:
            v1_ik = ak / dist_val
            # a_k^2 = 1 + gk_old^T S_k^-1 gk_old
            a_sq = ak * ak
            v2_ik = (1.0 + np.sqrt(a_sq * dist_val*dist_val)) / a_sq

            # Accumulate partial sums for gamma_k, mu_k, etc.
            zik = gamma_matrix[i, k]
            sum_zk[k]      += zik
            sum_zk_v1[k]   += zik * v1_ik
            sum_zk_v2[k]   += zik * v2_ik
            sum_zk_x[k]    += zik * xi
            sum_zk_v1_x[k] += zik * v1_ik * xi

            # Now partial sums for Sigma:
            # old formula from MATLAB:
            #   [z_ik * v1_ik * (x_i - mu_k_old)(x_i - mu_k_old)^T]  -  [gk_old*gk_old^T * z_ik*v2_ik]
            # stored in S_k_num
            outer_xdiff = np.outer(diff, diff)
            S_k_num[k] += zik * v1_ik * outer_xdiff  -  np.outer(gk_old, gk_old)* (zik * v2_ik)

            # And the denominator piece:
            #   z_ik*v1_ik * diff^T * Sk_inv * diff  -  (gk_old^T Sk_inv gk_old) * z_ik*v2_ik
            S_k_den[k] += zik * v1_ik * (diff @ Sk_inv @ diff)  \
                          -  (gk_old @ Sk_inv @ gk_old) * (zik * v2_ik)

    # Now finish updates for each cluster k, using the partial sums:
    for k in range(K):
        # (1) Update gamma_k:
        denom_k = sum_zk_v2[k] - (sum_zk[k]**2 / sum_zk_v1[k])
        if abs(denom_k) < 1e-14:
            denom_k = 1e-14 * np.sign(denom_k if denom_k != 0 else 1)
        
        # part_k = sum_zk_x[k] - sum_zk[k]*( sum_zk_v1_x[k]/sum_zk_v1[k] )
        part_k = sum_zk_x[k] - sum_zk[k] * (sum_zk_v1_x[k]/sum_zk_v1[k])
        gk_new = part_k / denom_k  # shape (p,)

        # (2) Update mu_k:
        mu_k_new = (sum_zk_v1_x[k] - gk_new*sum_zk[k]) / sum_zk_v1[k]

        # (3) Update Sigma_k:
        # numerator = p * S_k_num[k]
        # denominator = S_k_den[k], then Sigma_k_new = numerator / denominator
        # again clip if near zero
        denom_sigma_k = S_k_den[k]
        if abs(denom_sigma_k) < 1e-14:
            denom_sigma_k = 1e-14 * np.sign(denom_sigma_k if denom_sigma_k != 0 else 1)
        Sigma_k_new = (p * S_k_num[k]) / denom_sigma_k

        alpha_new[k] = (mu_k_new, Sigma_k_new,gk_new)

    return alpha_new

def compute_log_pdf_skewlaplace(X, mu_k, Sigma_k, gamma_k):
    """
    Computes the log PDF of a multivariate skew-Laplace distribution for given parameters.

    Parameters:
    ----------
    X : array-like, shape (N, p)
        Data points where the density needs to be evaluated.
    mu_k : array-like, shape (p,)
        Mean vector of the distribution.
    Sigma_k : array-like, shape (p, p)
        Covariance matrix (must be positive-definite).
    gamma_k : array-like, shape (p,)
        Skewness vector.

    Returns:
    -------
    log_probs : array, shape (N,)
        Log-density values at each data point.
    """
    N, p = X.shape

    # Invert & logdet Sigma_k
    sign, logdetS = np.linalg.slogdet(Sigma_k)  # sign should be +1 for PD matrix
    invS_k = np.linalg.solve(Sigma_k, np.eye(p))  # More stable than np.linalg.inv()

    # Compute a_k = sqrt(1 + gamma_k.T @ invS_k @ gamma_k)
    gamma_invS_gamma = gamma_k @ invS_k @ gamma_k  # Scalar
    a_k = np.sqrt(1.0 + gamma_invS_gamma)

    # Normalizing constant in log domain
    logC_k = (-0.5 * logdetS
              - (p * np.log(2.0)
                 + (p - 1) * 0.5 * math.log(math.pi)
                 + np.log(a_k)
                 + np.log(gamma_func((p + 1) / 2.0))))

    # Compute Mahalanobis distance
    Xm = X - mu_k  # shape (N, p)
    dist_sq = np.einsum('ij,jk,ik->i', Xm, invS_k, Xm)  # shape (N,)
    dist_val = np.sqrt(np.clip(dist_sq, 1e-14, None))  # Ensure no sqrt(0) errors

    # Compute linear term
    lin_term = np.einsum('ij,j->i', Xm @ invS_k, gamma_k)  # shape (N,)

    # Compute exponent
    exponent = -a_k * dist_val + lin_term

    log_probs = logC_k + exponent
    return log_probs
    

class SkewLaplaceMix(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True):
        super().__init__(n_clusters=n_clusters, EM_type="Soft", mixture_type="identical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.initialization=initialization
        self.family = "skew_laplace"
        
    def _log_pdf_skewlaplace(self,X,alphas):
        """
        Returns an array of shape (N, K) where entry (i, k) is
        log [ f_k( X[i] ) ], the log of the k-th Skew-Laplace pdf at X[i].
    
        X: shape (N, p)
        alphas: list of length K, each is {"mu": (p,), "Sigma": (p,p), "gamma": (p,)}.
        """
        N, p = X.shape
        K = len(alphas)
        log_probs = np.zeros((N, K))
    
        for k in range(K):
            mu_k = alphas[k][0]        # shape (p,)
            Sigma_k = alphas[k][1]  # shape (p,p)
            gamma_k = alphas[k][2]  # shape (p,)
    
            log_probs[:, k] = compute_log_pdf_skewlaplace(X, mu_k, Sigma_k, gamma_k)
    
        return log_probs
        
    
    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        return self._log_pdf_skewlaplace(X,alpha) + np.log(pi)
    
    def fit(self,sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2))
        np.random.seed(0)
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = k_means_init_skewlaplace(self.data, self.k)
        elif self.initialization == "gmm":
            self.pi_not, self.alpha_not = gmm_init_skewlaplace(self.data, self.k)
        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not
        pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(self.data,self.pi_temp,self.alpha_temp,estimate_alphas_skewlaplace)
        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.cluster = log_gamma_new.argmax(axis=1)
        self.gamma_temp_ar = np.exp(log_gamma_new)
        self.log_likelihood_new = log_likelihood_new
        if self.verbose:
            print("Mixtures of Skew-Laplace Fitting Done Successfully")
        end_time = time.time()
        self.execution_time=end_time-start_time
    
    
    
    
    
    def get_params(self):
        #print("The estimated pi values are ", self.pi_new)
        #print("The estimated alpha values are ", self.alpha_new)

        return self.pi_new, self.alpha_new

    def predict(self):
        return self.cluster

    def predict_new(self, x):
        x=self._process_data(x)
        data_lol = x.tolist()
        cluster, _ = mixture_clusters(self.gamma_temp_ar, data_lol)

        return cluster


    def responsibilities(self):

        return self.gamma_temp_ar

    def n_iter(self):
        return len(self.log_likelihoods)
    
    def get_info_mat(self):
        
        IM, SE = compute_empirical_info_and_se(self)
        
        return IM, SE
        
