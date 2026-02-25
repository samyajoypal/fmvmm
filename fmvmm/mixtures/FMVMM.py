from scipy.optimize import approx_fprime
import numpy as np
from fmvmm.utils.utils_fmm import (fmm_kmeans_init,fmm_gmm_init, fmm_loglikelihood, fmm_responsibilities,
                             fmm_pi_estimate, fmm_estimate_alphas, fmm_aic, fmm_bic, fmm_icl)


from fmvmm.utils.utils_mixture import (mixture_clusters)
import math

import itertools

from fmvmm.distributions import multivariate_skewnorm as mvsn
from fmvmm.distributions import multivariate_genhyperbolic as mghp
from fmvmm.distributions import multivariate_genskewt as mgst
from fmvmm.distributions import multivariate_hyperbolic as mvhb
from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_norminvgauss as mnig
from fmvmm.distributions import multivariate_t as mvt
from fmvmm.distributions import multivariate_vargamma as mvvg
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.distributions import multivariate_skew_t_smsn as mvst
from fmvmm.distributions import multivariate_skewnorm_cont as msnc
from fmvmm.distributions import multivariate_skewslash as mssl
from fmvmm.distributions import multivariate_slash as msl
import traceback
from scipy.special import logsumexp
from fmvmm.mixtures._base import BaseMixture
from fmvmm.utils.utils_dmm import hard_assignments, mixture_counts, mixture_proportions_info
from fmvmm.mixtures.mixmgh import approx_hessian_scipy
import warnings
warnings.filterwarnings('ignore')

dist_map = {"mvsn": mvsn, "mghp": mghp, "mgst": mgst, "mvhb": mvhb,
            "mvn": mvn, "mnig": mnig, "mvt": mvt, "mvvg": mvvg, "mvsl": mvsl,
            "mvst": mvst, "msnc": msnc, "mssl": mssl, "msl": msl}

all_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg",
            "mvsl", "mvst", "msnc","mssl","msl"]

def convert_to_numpy(tuples_list):
    """
    Given a list of tuples, convert any integer or float inside the tuples
    into a NumPy array of shape (1,).
    """
    def convert_item(item):
        if isinstance(item, (int, float)):
            return np.array([item])
        elif isinstance(item, tuple):
            return tuple(convert_item(sub_item) for sub_item in item)
        elif isinstance(item, list):  # If lists are inside tuples, handle them too
            return [convert_item(sub_item) for sub_item in item]
        else:
            return item  # Keep other types unchanged

    return [tuple(convert_item(item) for item in tpl) for tpl in tuples_list]


# def compute_info_individual_fmvmm(dist_comp, params_init, X, epsilon=1e-6):
#     """
#     Use SciPy to approximate the Hessian of the negative log-likelihood
#     for the GH mixture, then invert to get the empirical info matrix.
#
#     Returns: I_e, Cov, SE
#     """
#     IM = dist_comp.info_mat(X, *params_init)
#
#     # Flatten final parameters
#     n, p = X.shape
#     param_shapes = [np.array(p).shape for p in params_init]
#     theta_hat = flatten_params(params_init)
#
#     d = len(theta_hat)
#
#
#
#     return IM, len(theta_hat)

def compute_info_individual_fmvmm(dist_comp, params_init, X, epsilon=1e-6,
                                  prefer_scores=True, ridge=1e-8):
    """
    Returns
    -------
    IM : (d,d)
    param_len : int
    se : (d,)  (optional, convenient)
    cov : (d,d)
    meta : dict (debug info)
    """
    X = np.asarray(X)
    n, p = X.shape

    # ---------- 1) Try score-based OPG if available ----------
    if prefer_scores and hasattr(dist_comp, "score_mat") and callable(getattr(dist_comp, "score_mat")):
        S = dist_comp.score_mat(X, *params_init)  # (n,d)
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[0] != n:
            raise ValueError(f"score_mat must return shape (n,d). Got {S.shape}")
        S -= S.mean(axis=0, keepdims=True)   # << important

        IM = S.T @ S
        IM = 0.5 * (IM + IM.T) + ridge * np.eye(IM.shape[0])

        try:
            cov = np.linalg.inv(IM)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(IM)

        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return IM, IM.shape[0], se, cov, {"source": "scores_opg", "n": n, "p": p}

    # ---------- 2) Fallback: info_mat ----------
    IM = dist_comp.info_mat(X, *params_init)
    IM = np.asarray(IM, float)
    IM = 0.5 * (IM + IM.T) + ridge * np.eye(IM.shape[0])

    try:
        cov = np.linalg.inv(IM)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(IM)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return IM, IM.shape[0], se, cov, {"source": "info_mat", "n": n, "p": p}

def organize_data_by_clusters(data, cluster_predictions):
    # Get the unique cluster labels
    unique_clusters = np.unique(cluster_predictions)

    # Initialize a list to store data points for each cluster
    data_by_clusters = [np.empty((0, data.shape[1])) for _ in unique_clusters]

    # Iterate over each data point and its cluster prediction
    for point, cluster in zip(data, cluster_predictions):
        # Append the data point to the corresponding cluster
        data_by_clusters[cluster] = np.vstack((data_by_clusters[cluster], point))

    return data_by_clusters


def flatten_params(params):
    """
    Flatten parameters into a single vector.

    Parameters:
        params (tuple): The parameters.

    Returns:
        flat_params (ndarray): The flattened parameters.
    """
    return np.array(np.concatenate([p.flatten() if isinstance(p, np.ndarray) else np.array([p]) for p in params]), dtype=float)


def reshape_params(flat_params, param_shapes):
    """
    Reshape a flattened parameter vector into original shapes.

    Parameters:
        flat_params (ndarray): The flattened parameters.
        param_shapes (list): The shapes of the original parameters.

    Returns:
        params (tuple): The reshaped parameters.
    """
    params = []
    start = 0
    for shape in param_shapes:
        end = start + np.prod(shape)
        params.append(np.reshape(flat_params[start:end], shape))
        start = end
    return tuple(params)

def ensure_positive_diagonal(sum_outer):
    """
    Ensure that the matrix has positive diagonal elements.

    Parameters:
        sum_outer (ndarray): The matrix. Shape (num_params, num_params).

    Returns:
        sum_outer_pos (ndarray): The matrix with positive diagonal elements.
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(sum_outer)

    # Adjust eigenvalues to ensure positivity
    eigenvalues[eigenvalues < 0] = 1e-6  # Set negative eigenvalues to a small positive value

    # Reconstruct the matrix
    sum_outer_pos = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))

    return sum_outer_pos

# def project_to_psd(A, eps=1e-8):
#     A = 0.5 * (A + A.T)
#     w, V = np.linalg.eigh(A)
#     w = np.maximum(w, eps)
#     return (V * w) @ V.T

def project_to_psd(A, eps=1e-8, max_tries=3):
    """
    Robust PSD projection with ridge fallback.
    """

    A = 0.5 * (A + A.T)

    for _ in range(max_tries):
        try:
            w, V = np.linalg.eigh(A)
            w = np.maximum(w, eps)
            return (V * w) @ V.T
        except np.linalg.LinAlgError:
            # inflate diagonal and retry
            A = A + 10 * eps * np.eye(A.shape[0])

    # ultimate fallback: diagonal approximation
    diag = np.maximum(np.diag(A), eps)
    return np.diag(diag)

def pi_info_constrained(pi, N, eps=1e-12):
    """
    Constrained (simplex-aware) information for mixture weights
    using logits eta_j = log(pi_j/pi_k), j=1..k-1.

    Returns:
      I_eta      : (k-1)x(k-1) information in eta coords
      Cov_eta    : inverse (pseudo-inverse if needed)
      Cov_pi     : approximate covariance for pi (k x k) via delta method
    """
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    pi = np.clip(pi, eps, 1 - eps)
    pi = pi / pi.sum()

    p = pi[:k-1]                 # length k-1
    I_eta = N * (np.diag(p) - np.outer(p, p))  # (k-1)x(k-1)

    # invert in eta space
    I_eta = 0.5 * (I_eta + I_eta.T)
    try:
        Cov_eta = np.linalg.inv(I_eta)
    except np.linalg.LinAlgError:
        Cov_eta = np.linalg.pinv(I_eta)

    # Jacobian J = d pi / d eta  (k x (k-1))
    # For j<k: d pi_j / d eta_m = pi_j*(1{j=m} - pi_m)
    # For k:   d pi_k / d eta_m = -pi_k*pi_m
    J = np.zeros((k, k-1))
    for m in range(k-1):
        for j in range(k-1):
            J[j, m] = pi[j] * ((1.0 if j == m else 0.0) - pi[m])
        J[k-1, m] = -pi[k-1] * pi[m]

    Cov_pi = J @ Cov_eta @ J.T
    return I_eta, Cov_eta, Cov_pi


def ecdf_from_log_density(log_density):
    """
    Compute the empirical cumulative distribution function (ECDF) from log-density values.

    Parameters:
        log_density (numpy.ndarray): The array of log-density values.

    Returns:
        numpy.ndarray: The empirical cumulative distribution function values.
    """
    # Convert log-density to density by exponentiation
    density = np.exp(log_density)

    # Sort the density values in ascending order
    sorted_density = np.sort(density)

    # Compute the cumulative sum of the sorted density values
    cumulative_sum = np.cumsum(sorted_density)

    # Normalize the cumulative sum to get ECDF values
    ecdf_values = cumulative_sum / np.sum(density)

    return ecdf_values


def fmm_sorted_lpdf_cdf(pi_temp,alpha_temp,data_lol,list_of_dist):
    n = len(data_lol)
    k = len(alpha_temp)
    dist_variables = [dist_map[list_of_dist[j]]
                           for j in range(len(list_of_dist))]
    dist_comb = list(
        itertools.combinations(dist_variables, k))[0]
    log_likelihood_values_temp = []
    for c in range(n):
        try:
            log_likelihood_old_temp = math.log(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        except:
            log_likelihood_old_temp = math.log1p(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        log_likelihood_values_temp.append(log_likelihood_old_temp)

    sorted_lists = sorted(zip(log_likelihood_values_temp, data_lol),reverse=False)
    sorted_lpdf, sorted_data = zip(*sorted_lists)


    cdfs=ecdf_from_log_density(log_likelihood_values_temp)
    return sorted_lpdf,cdfs

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
        if any(np.isnan(temp)) or any(np.isinf(temp)):
            indices_to_remove.append(i)

    for i, inner_list in enumerate(list_of_lists):
        if any(val == 0 for val in inner_list):
            indices_to_remove.append(i)


    # Remove sub-lists and tuples using the stored indices
    if len(indices_to_remove)>0:
        list_of_lists = [list_of_lists[i] for i in range(len(list_of_lists)) if i not in indices_to_remove]
        list_of_list_of_tuples = [list_of_list_of_tuples[i] for i in range(len(list_of_list_of_tuples)) if i not in indices_to_remove]

    return list_of_lists, list_of_list_of_tuples, indices_to_remove

def remove_elements_by_index(my_list, index_list):
    if len(index_list)>0:
        return [item for i, item in enumerate(my_list) if i not in index_list]
    else:
        return my_list

def get_dist_names(dist_comb):
    return [str(c.__name__) for c in dist_comb]




class fmvmm(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001, list_of_dist=all_dist,specific_comb=False,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True, debug = False):
        super().__init__(n_clusters=n_clusters, EM_type="Hard", mixture_type="nonidentical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.list_of_dist = list_of_dist
        self.specific_comb=specific_comb
        self.dist_variables = [dist_map[list_of_dist[j]]
                               for j in range(len(list_of_dist))]
        self.specific_comb=specific_comb
        if self.specific_comb==True:
            self.dist_combs = list(
                itertools.combinations(self.dist_variables, self.k))
        else:
            self.dist_combs = list(
                itertools.combinations_with_replacement(self.dist_variables, self.k))
        self.initialization=initialization
        self.debug = debug

    def _log_pdf_non_identical(self,X,alphas,dist_comb):
        N,p=X.shape
        k=len(alphas)
        probs=np.empty((N, k))
        for j in range(k):
            alpha=alphas[j]
            probs[:, j]=dist_comb[j].logpdf(X,*alpha)
        return probs

    def _estimate_weighted_log_prob_nonidentical(self, X, alpha, pi, dist_comb):
        return self._log_pdf_non_identical(X,alpha,dist_comb) + np.log(pi)

    def fit(self,sample):
        self.data = self._process_data(sample)
        self.n=len(sample)
        # self.sample=self._process_data(sample)
        self.list_aic = []
        self.list_bic = []
        self.list_icl = []
        self.list_pi = []
        self.list_alpha = []
        self.list_cluster = []
        self.list_log_likelihood = []
        self.list_all_log_likelihood = []
        self.list_gamma_matrix = []
        self.not_worked_dist=[]
        self.worked_dist=[]
        for l in range(len(self.dist_combs)):
            try:

                if self.initialization=="kmeans":

                    self.pi_not, self.alpha_not = fmm_kmeans_init(
                        self.data, self.k, self.dist_combs[l])
                    self.alpha_temp = self.alpha_not
                    self.pi_temp = self.pi_not
                else:
                    self.pi_not, self.alpha_not = fmm_gmm_init(
                        self.data, self.k, self.dist_combs[l])
                    self.alpha_temp = self.alpha_not
                    self.pi_temp = self.pi_not
                pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(self.data,self.pi_temp,self.alpha_temp,fmm_estimate_alphas,dist_comb=self.dist_combs[l])
                self.list_aic.append(
                    fmm_aic(alpha_new, log_likelihood_new, self.dist_combs[l] ))
                self.list_bic.append(
                    fmm_bic(alpha_new, log_likelihood_new,self.dist_combs[l], self.n))
                self.list_icl.append(
                    fmm_icl(alpha_new, log_likelihood_new,self.dist_combs[l], self.n, np.exp(log_gamma_new)))
                self.list_pi.append(pi_new)
                # self.list_alpha.append(convert_to_numpy(alpha_new))
                self.list_alpha.append(alpha_new)
                self.list_cluster.append(log_gamma_new.argmax(axis=1))
                self.list_log_likelihood.append(log_likelihood_new)
                self.list_gamma_matrix.append(np.exp(log_gamma_new))
                self.worked_dist.append(self.dist_combs[l])
                self.list_all_log_likelihood.append(self.log_likelihoods)
                if self.verbose:
                    print("distribution fitted", get_dist_names(self.dist_combs[l]))
            except:
                if self.debug:
                    traceback.print_exc()
                    print("Error received while running,",self.dist_combs[l])
                self.not_worked_dist.append(self.dist_combs[l])
                pass
        self.list_pi,self.list_alpha,self.nan_ind=remove_nan_tuples(self.list_pi,self.list_alpha)
        self.list_aic=remove_elements_by_index(self.list_aic,self.nan_ind)
        self.list_bic=remove_elements_by_index(self.list_bic, self.nan_ind)
        self.list_icl=remove_elements_by_index(self.list_icl, self.nan_ind)
        self.list_cluster=remove_elements_by_index(self.list_cluster, self.nan_ind)
        self.list_log_likelihood=remove_elements_by_index(self.list_log_likelihood, self.nan_ind)
        self.list_gamma_matrix=remove_elements_by_index(self.list_gamma_matrix, self.nan_ind)
        self.not_worked_dist.extend(w for w in [self.worked_dist[h] for h in self.nan_ind])
        self.worked_dist=remove_elements_by_index(self.worked_dist, self.nan_ind)
        self.list_all_log_likelihood=remove_elements_by_index(self.list_all_log_likelihood, self.nan_ind)
        self.fitted=True
        if self.verbose:
            print("Model fitted successfully")

    def get_params(self):
        return self.list_pi, self.list_alpha

    def predict(self):
        return self.list_cluster

    def predict_new(self, X):
        # data_lol = x.values.tolist()
        cluster_all = []
        for l in range(len(self.dist_combs)):
            # cluster, _ = mixture_clusters(self.gamma_matrix[l], data_lol)
            cluster_all.append(self._predict(X))

        return cluster_all


    def best_mixture(self):
        best_mix = self.worked_dist[np.argmin(self.list_bic)]
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

    def get_maximum_likelihood(self):
        return self.list_log_likelihood
    def get_all_likelihood(self):
        return self.list_all_log_likelihood


    # def get_info_mat(self):
    #     """
    #     Assemble the total Fisher information matrix for each 'worked_dist' mixture
    #     distribution, combining the mixture-proportions info block and the
    #     parameter info blocks for each component.
    #
    #     Returns:
    #       mixture_info_mats: list of 2D arrays, one per mixture distribution
    #       mixture_ses:       1D array of standard errors (concatenated across mixtures)
    #     """
    #     mixture_ses = []
    #     mixture_info_mats = []
    #
    #     # Loop over each mixture distribution
    #     for a, mix in enumerate(self.worked_dist):
    #         # mix is the list of components for this mixture
    #         # e.g. mix = [comp_1, comp_2, ..., comp_k]
    #         k = len(mix)  # number of components
    #
    #         # Extract data assigned to each component, etc.
    #         cwise_data = organize_data_by_clusters(np.array(self.data), self.list_cluster[a])
    #         mle_params = self.list_alpha[a]  # parameters for each component
    #         N_j, N = mixture_counts(self.list_gamma_matrix[a], mode="hard")
    #
    #         # 1) Mixture proportion info
    #         I_pi, I_pi_inv = mixture_proportions_info(self.list_pi[a], N_j)
    #         # I_pi, I_pi_inv are k×k
    #
    #         # 2) Component-level Fisher infos
    #         #    We'll collect the Fisher info for each component and store
    #         #    their inverses in I_inv_blocks for the final block assembly.
    #         mix_info = []
    #         param_dims = []    # dimension of each component's parameter vector
    #         I_inv_blocks = []
    #
    #         for b, mix_dist in enumerate(mix):
    #             # Compute Fisher info for the b-th component
    #             fisher_info, param_len = compute_info_individual_fmvmm(
    #                 mix_dist,   # log PDF function
    #                 mle_params[b],     # parameters for that component
    #                 cwise_data[b],     # data assigned to that component
    #                 epsilon=0.06
    #             )
    #             mix_info.append(fisher_info)
    #             param_dims.append(param_len)
    #
    #             # Invert it (or pseudo-invert if singular)
    #             try:
    #                 inv_block = np.linalg.inv(fisher_info)
    #             except np.linalg.LinAlgError:
    #                 inv_block = np.linalg.pinv(fisher_info)
    #
    #             I_inv_blocks.append(inv_block)
    #
    #         # 3) Now compute the total dimension:
    #         #    - k for the mixture proportions
    #         #    - plus the sum of parameter dims across all components
    #         total_param_dim = sum(param_dims)
    #         big_dim = k + total_param_dim
    #
    #         # Allocate the overall info matrix and its inverse
    #         I_total = np.zeros((big_dim, big_dim))
    #         I_total_inv = np.zeros((big_dim, big_dim))
    #
    #         # 4) Insert mixture proportion info in the top-left k×k block
    #         I_total[:k, :k] = I_pi
    #         I_total_inv[:k, :k] = I_pi_inv
    #
    #         # 5) Insert each component's block
    #         row_start = k  # begin filling right below/after the k×k block
    #         for j in range(k):
    #             param_dim_j = param_dims[j]
    #             row_end = row_start + param_dim_j
    #
    #             # Place the j-th component's fisher info block
    #             I_total[row_start:row_end, row_start:row_end] = mix_info[j]
    #             I_total_inv[row_start:row_end, row_start:row_end] = I_inv_blocks[j]
    #
    #             row_start = row_end  # advance to the next block
    #
    #         # 6) Compute standard errors from the diagonal of the total inverse
    #         #    (Ensure positive diagonals so we don't end up with sqrt of negative)
    #         var_diag = np.diag(ensure_positive_diagonal(I_total_inv))
    #         se_total = np.sqrt(var_diag)
    #
    #         # 7) Store the results
    #         mixture_ses.append(se_total)      # collect standard errors
    #         mixture_info_mats.append(I_total) # collect the big Fisher matrix
    #
    #     return mixture_info_mats, mixture_ses

    def get_info_mat(self):
        """
        Compute (blockwise) Fisher information matrices and standard errors
        for each worked mixture.

        Uses:
          - constrained mixture-weight information (pi_info_constrained)
          - PSD-projected component Fisher blocks
          - blockwise inversion (no huge pinv)

        Returns
        -------
        mixture_info_mats : list of 2D ndarrays
            Block-diagonal Fisher matrices
        mixture_ses : list of 1D ndarrays
            Concatenated SE vectors
        """

        mixture_ses = []
        mixture_info_mats = []

        for a, mix in enumerate(self.worked_dist):

            k = len(mix)

            # -----------------------------
            # Data & params
            # -----------------------------
            cwise_data = organize_data_by_clusters(
                np.array(self.data),
                self.list_cluster[a]
            )

            mle_params = self.list_alpha[a]
            N_j, N = mixture_counts(self.list_gamma_matrix[a], mode="hard")

            # ============================================================
            # (1) Mixture weights (constrained)
            # ============================================================
            _, _, Cov_pi = pi_info_constrained(self.list_pi[a], N)
            I_pi = np.linalg.pinv(Cov_pi)   # k x k

            var_pi = np.maximum(np.diag(Cov_pi), 1e-12)
            se_parts = [np.sqrt(var_pi)]

            # ============================================================
            # (2) Component blocks
            # ============================================================
            mix_info = []
            param_dims = []

            for b, mix_dist in enumerate(mix):

                fisher_info, param_len,_,_,_ = compute_info_individual_fmvmm(
                    mix_dist,
                    mle_params[b],
                    cwise_data[b],
                    epsilon=0.06
                )

                # enforce symmetry + PSD
                fisher_info = project_to_psd(fisher_info, eps=1e-8)

                try:
                    inv_block = np.linalg.inv(fisher_info)
                except np.linalg.LinAlgError:
                    inv_block = np.linalg.pinv(fisher_info)

                var_b = np.maximum(np.diag(inv_block), 1e-12)
                se_parts.append(np.sqrt(var_b))

                mix_info.append(fisher_info)
                param_dims.append(param_len)

            # ============================================================
            # (3) Assemble block-diagonal Fisher (optional, for output)
            # ============================================================
            big_dim = k + sum(param_dims)
            I_total = np.zeros((big_dim, big_dim))

            # pi block
            # I_total[:k, :k] = np.linalg.pinv(Cov_pi)
            I_total[:k, :k] = I_pi

            row = k
            for j in range(k):
                d = param_dims[j]
                I_total[row:row+d, row:row+d] = mix_info[j]
                row += d

            # ============================================================
            # (4) Store
            # ============================================================
            mixture_ses.append(np.concatenate(se_parts))
            mixture_info_mats.append(I_total)

        return mixture_info_mats, mixture_ses

    def get_info_mat_soft(self):
        """
        Observed-likelihood OPG information using weighted per-observation scores.

        For component h:
            u_i,h = tau_{ih} * s_{ih}
            I_h = sum_i u_i,h u_i,h^T

        Returns
        -------
        mixture_info_mats : list of 2D ndarrays
        mixture_ses : list of 1D ndarrays
        """

        mixture_ses = []
        mixture_info_mats = []

        for a, mix in enumerate(self.worked_dist):

            k = len(mix)
            X = np.asarray(self.data)
            tau = self.list_gamma_matrix[a]      # (n, k)
            pi = self.list_pi[a]
            n = X.shape[0]

            # ============================================================
            # (1) Mixture weights (simplex-aware)
            # ============================================================
            _, _, Cov_pi = pi_info_constrained(pi, n)
            I_pi = np.linalg.pinv(Cov_pi)

            se_parts = [np.sqrt(np.maximum(np.diag(Cov_pi), 1e-12))]

            mix_info = []
            param_dims = []

            # ============================================================
            # (2) Component blocks (soft OPG)
            # ============================================================
            for h, mix_dist in enumerate(mix):

                # raw per-observation scores: (n, d_h)
                S = mix_dist.score_mat(X, *self.list_alpha[a][h])
                S = np.asarray(S, float)

                if S.ndim != 2:
                    raise ValueError("score_mat must return (n,d) matrix")

                # weight by tau
                W = tau[:, h][:, None]          # (n,1)
                S_w = W * S                     # (n,d)

                # OPG
                I_h = S_w.T @ S_w
                I_h = 0.5 * (I_h + I_h.T)
                I_h += 1e-8 * np.eye(I_h.shape[0])
                I_h = project_to_psd(I_h, eps=1e-8)

                try:
                    cov_h = np.linalg.inv(I_h)
                except np.linalg.LinAlgError:
                    cov_h = np.linalg.pinv(I_h)

                se_h = np.sqrt(np.maximum(np.diag(cov_h), 1e-12))

                mix_info.append(I_h)
                param_dims.append(I_h.shape[0])
                se_parts.append(se_h)

            # ============================================================
            # (3) Assemble block-diagonal Fisher
            # ============================================================
            big_dim = k + sum(param_dims)
            I_total = np.zeros((big_dim, big_dim))

            # pi block
            I_total[:k, :k] = I_pi

            row = k
            for h in range(k):
                d = param_dims[h]
                I_total[row:row+d, row:row+d] = mix_info[h]
                row += d

            mixture_info_mats.append(I_total)
            mixture_ses.append(np.concatenate(se_parts))

        return mixture_info_mats, mixture_ses

    def get_top_mixtures(self,n_top=10):
        wrk_lst=[]
        for i in range(len(self.worked_dist)):
            wrk_lst.append([str(self.worked_dist[i][j].__name__) for j in range(len(self.worked_dist[i]))])

        # Combine the two lists into a list of tuples
        combined = list(zip(self.list_bic, wrk_lst))

        # Sort the list of tuples based on bic values
        sorted_combined = sorted(combined)

        return sorted_combined[:n_top]
