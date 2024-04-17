from scipy.optimize import approx_fprime
import numpy as np
from fmvmm.utils.utils_fmm import (fmm_kmeans_init,fmm_gmm_init, fmm_loglikelihood, fmm_responsibilities,
                             fmm_pi_estimate, fmm_estimate_alphas, fmm_aic, fmm_bic)


from fmvmm.utils.utils_mixture import (mixture_clusters)
import math

import itertools

from fmvmm.mvem.stats import multivariate_skewnorm as mvsn
from fmvmm.mvem.stats import multivariate_genhyperbolic as mghp
from fmvmm.mvem.stats import multivariate_genskewt as mgst
from fmvmm.mvem.stats import multivariate_hyperbolic as mvhb
from fmvmm.mvem.stats import multivariate_norm as mvn
from fmvmm.mvem.stats import multivariate_norminvgauss as mnig
from fmvmm.mvem.stats import multivariate_t as mvt
from fmvmm.mvem.stats import multivariate_vargamma as mvvg
import traceback
from scipy.special import logsumexp
from fmvmm.mixtures._base import BaseMixture

dist_map = {"mvsn": mvsn, "mghp": mghp, "mgst": mgst, "mvhb": mvhb,
            "mvn": mvn, "mnig": mnig, "mvt": mvt, "mvvg": mvvg}

all_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg"]

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
    return np.concatenate([p.flatten() for p in params])

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

def logpdf_wrapper(X,ind, flat_params, logpdf_func, param_shapes):
    """
    Wrapper function to compute the log-pdf for all data points using flattened parameters.

    Parameters:
        X (ndarray): The data points. Shape (N, p).
        flat_params (ndarray): Flattened parameters.
        logpdf_func (callable): The log-pdf function.
        param_shapes (list): The shapes of the original parameters.

    Returns:
        logpdf_values (ndarray): The log-pdf values for all data points.
    """
    params = reshape_params(flat_params, param_shapes)
    return logpdf_func(X, *params)[ind]

def score_vectors(logpdf_func, X, flat_params, param_shapes, epsilon=1e-6):
    """
    Compute the score vectors (gradient of log likelihood with respect to parameters)
    for each data point in X.

    Parameters:
        logpdf_func (callable): The log probability density function.
        X (ndarray): The data points. Shape (N, p).
        flat_params (ndarray): The flattened parameters.
        param_shapes (list): The shapes of the original parameters.
        epsilon (float): Step size for finite differences.

    Returns:
        scores (ndarray): The score vectors for each data point. Shape (N, num_params).
    """
    num_params = len(flat_params)
    N, p = X.shape
    scores = np.zeros((N, num_params))

    # Compute score vector for each data point
    for i in range(N):
        # Compute gradient of logpdf for data point i
        gradient = approx_fprime(flat_params,
                                  lambda flat_params: logpdf_wrapper(X,i, flat_params, logpdf_func, param_shapes),
                                  epsilon=epsilon)
        scores[i, :] = gradient

    return scores

def sum_outer_products(scores):
    """
    Compute the sum of the outer products of the score vectors.

    Parameters:
        scores (ndarray): The score vectors for each data point. Shape (N, num_params).

    Returns:
        sum_outer (ndarray): The sum of the outer products of the score vectors. Shape (num_params, num_params).
    """
    N, num_params = scores.shape
    sum_outer = np.zeros((num_params, num_params))

    # Compute sum of outer products
    for i in range(N):
        sum_outer += np.outer(scores[i], scores[i])

    return sum_outer

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

def compute_expected_info(logpdf_func, params_init, X, epsilon=1e-6):
    param_shapes = [p.shape for p in params_init]
    params_init_flat = flatten_params(params_init)
    scores=score_vectors(logpdf_func, X, params_init_flat, param_shapes, epsilon)
    # scores=score_vectors(logpdf_func, X, params_init, epsilon)
    # print(scores)
    sum_outer = sum_outer_products(scores)

    return sum_outer

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
    def __init__(self,n_clusters,tol=0.0001, list_of_dist=all_dist,specific_comb=False,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True):
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
        self.data=sample
        self.n=len(sample)
        # self.sample=self._process_data(sample)
        self.list_aic = []
        self.list_bic = []
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
                    fmm_aic(self.k, alpha_new, log_likelihood_new))
                self.list_bic.append(
                    fmm_bic(self.k, alpha_new, log_likelihood_new, self.n))
                self.list_pi.append(pi_new)
                self.list_alpha.append(alpha_new)
                self.list_cluster.append(log_gamma_new.argmax(axis=1))
                self.list_log_likelihood.append(log_likelihood_new)
                self.list_gamma_matrix.append(np.exp(log_gamma_new))
                self.worked_dist.append(self.dist_combs[l])
                self.list_all_log_likelihood.append(self.log_likelihoods)
                if self.verbose:
                    print("distribution fitted", get_dist_names(self.dist_combs[l]))
            except:
                traceback.print_exc()
                #print("Error received while running,",self.dist_combs[l])
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

    def get_maximum_likelihood(self):
        return self.list_log_likelihood
    def get_all_likelihood(self):
        return self.list_all_log_likelihood

    def get_standard_error(self):
        mixture_ses=[]
        for a,mix in enumerate(self.worked_dist):
            mix_se=[]
            cwise_data=organize_data_by_clusters(np.array(self.sample),self.list_cluster[a])
            mle_params=self.list_alpha[a]
            for b,mix_dist in enumerate(mix):
                # print(cwise_data[b])
                # Calculate the Hessian at the MLE
                # hessian =  compute_hessian_scipy(mix_dist.loglike,mle_params[b],cwise_data[b],epsilon=1e-8)
                # Fisher information is the negative expectation of the Hessian
                # Calculate Fisher information
                # fisher_info = -1 *hessian
                fisher_info=compute_expected_info(mix_dist.logpdf,mle_params[b],cwise_data[b],epsilon=0.01)
                # print(fisher_info)
                try:
                    inv=np.linalg.inv(fisher_info)
                except:
                   inv=np.linalg.pinv(fisher_info)
                diag_inv=np.diag(ensure_positive_diagonal(inv))
                # diag_inv=np.diag(inv)
                # print(diag_inv)
                se=np.sqrt(diag_inv)

                mix_se.append(np.real(np.round(se,5)))
            mixture_ses.extend(mix_se)

        return mixture_ses

    def get_top_mixtures(self,n_top=10):
        wrk_lst=[]
        for i in range(len(self.worked_dist)):
            wrk_lst.append([str(self.worked_dist[i][j].__name__) for j in range(len(self.worked_dist[i]))])

        # Combine the two lists into a list of tuples
        combined = list(zip(self.list_bic, wrk_lst))

        # Sort the list of tuples based on bic values
        sorted_combined = sorted(combined)

        return sorted_combined[:n_top]        
