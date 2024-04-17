import numpy as np
import pandas as pd
from scipy.special import logsumexp
from abc import ABCMeta, abstractmethod
from fmvmm.utils import utils_mixture

class BaseMixture(metaclass=ABCMeta):

    def __init__(self, n_clusters, EM_type, mixture_type, tol, print_log_likelihood, max_iter, verbose):
        self.n_clusters = n_clusters
        self.EM_type = EM_type
        self.mixture_type = mixture_type
        self.tol = tol
        self.print_log_likelihood = print_log_likelihood
        self.max_iter = max_iter
        self.verbose = verbose
        self.fitted=False
        
    def _process_data(self,X):
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray) and len(X.shape) == 2:
            return X
        elif isinstance(X, list):
            if all(isinstance(row, list) for row in X):
                return np.array(X)
            else:
                raise ValueError("Input should be a list of lists for conversion to 2D NumPy array.")
        else:
            raise ValueError("Input should be a Pandas DataFrame, a 2D NumPy array, or a list of lists.")

    # @abstractmethod
    # def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
    #     pass

    # @abstractmethod
    # def _estimate_weighted_log_prob_nonidentical(self, X, alpha, pi, dist_comb):
    #     pass

    def _estimate_log_prob_resp_identical(self, X, alpha, pi):
        weighted_log_prob = self._estimate_weighted_log_prob_identical(
            X, alpha, pi)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_log_prob_resp_nonidentical(self, X, alpha, pi, dist_comb):
        weighted_log_prob = self._estimate_weighted_log_prob_nonidentical(
            X, alpha, pi, dist_comb)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _e_step(self, X, alpha, pi, dist_comb=None):
        if self.mixture_type == "identical":
            log_prob_norm, log_resp = self._estimate_log_prob_resp_identical(
                X, alpha, pi)
        elif self.mixture_type == "nonidentical":
            log_prob_norm, log_resp = self._estimate_log_prob_resp_nonidentical(
                X, alpha, pi, dist_comb)
        return np.sum(log_prob_norm), log_resp
    
    
    def _fit(self,X,pi_temp,alpha_temp,estimate_alphas_function,dist_comb=None,**kwargs):
        X=self._process_data(X)
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.log_likelihoods=[]
        stop=False
        while stop!=True:
            log_likelihood_old, log_gamma_temp = self._e_step(
                X, alpha_temp, pi_temp,dist_comb)
            if len(self.log_likelihoods)==0:
                self.log_likelihoods.append(log_likelihood_old)
            gamma_temp_ar=np.exp(log_gamma_temp)
            pi_new,alpha_new=self._m_step(gamma_temp_ar, X, estimate_alphas_function,dist_comb, **kwargs)
            log_likelihood_new, log_gamma_new = self._e_step(
                X, alpha_new, pi_new,dist_comb)
            self.log_likelihoods.append(log_likelihood_new)
            stop=self._stopping_criteria(self.log_likelihoods,self.tol,self.max_iter)
            log_like_diff = abs(log_likelihood_new-log_likelihood_old)/abs(log_likelihood_old)
            alpha_temp = alpha_new
            pi_temp = pi_new
            if self.print_log_likelihood==True:
                print("Loglikelihood:",log_likelihood_new, "Relative Difference:", log_like_diff )
            
        return pi_new,alpha_new, log_likelihood_new,log_gamma_new
    
    def _predict(self,X):
        X=self._process_data(X)
        if self.mixture_type == "identical":
            return self._estimate_weighted_log_prob_identical(X,self.alpha_new, self.pi_new).argmax(axis=1)
        elif self.mixture_type=="nonidentical":
            return self._estimate_weighted_log_prob_nonidentical(X,self.alpha_new, self.pi_new,self.dist_comb).argmax(axis=1)
            
            
    
    def _m_step(self,gamma_matrix,X,estimate_alphas_function,dist_comb,**kwargs):
        if self.EM_type=="Hard":
            cluster,data_cwise=utils_mixture.mixture_clusters(gamma_matrix, X)
            pi_new = np.bincount(cluster, minlength=gamma_matrix.shape[1]) / len(cluster)
            alpha_new=estimate_alphas_function(data_cwise,self.alpha_not,dist_comb,**kwargs)
        elif self.EM_type=="Soft":
            nj=gamma_matrix.sum(axis=0) + 10 * np.finfo(gamma_matrix.dtype).eps
            pi_new = nj / nj.sum()
            alpha_new=estimate_alphas_function(X,gamma_matrix,self.alpha_temp,**kwargs)

        return pi_new, alpha_new

    def _stopping_criteria(self, log_likelihoods, tol, n_iter_max):
        if len(log_likelihoods) == n_iter_max:
            return True
        elif abs(log_likelihoods[-1]-log_likelihoods[-2])/abs(log_likelihoods[-2]) <= tol:
            return True
        else:
            return False
    
    