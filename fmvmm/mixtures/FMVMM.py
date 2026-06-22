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
from fmvmm.mixtures.skewnormmix_smsn import (
    estimate_alphas_skewnormal, dmvSN, d_mixedmvSN
)
from fmvmm.mixtures.skewtmix_smsn import estimate_alphas_skewt
from fmvmm.mixtures.tmix_smsn import estimate_alphas_t
from fmvmm.mixtures.skewcontmix_smsn import (
    estimate_alphas_skewcn, dmvSNC, d_mixedmvSNC
)
from fmvmm.mixtures.skewslashmix_smsn import (
    estimate_alphas_skewslash,
    dmvSS as dmvSS_skewslash,
    d_mixedmvSS as d_mixedmvSS_skewslash,
)
from fmvmm.mixtures.slashmix_smsn import (
    estimate_alphas_slash,
    dmvSS as dmvSS_slash,
    d_mixedmvSS as d_mixedmvSS_slash,
)
from fmvmm.mixtures.skewlaplacemix import estimate_alphas_skewlaplace
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST
import warnings
warnings.filterwarnings('ignore')

dist_map = {"mvsn": mvsn, "mghp": mghp, "mgst": mgst, "mvhb": mvhb,
            "mvn": mvn, "mnig": mnig, "mvt": mvt, "mvvg": mvvg, "mvsl": mvsl,
            "mvst": mvst, "msnc": msnc, "mssl": mssl, "msl": msl}

all_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg",
            "mvsl", "mvst", "msnc","mssl","msl"]

soft_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg",
             "mvsl", "mvst", "msnc", "mssl", "msl"]
dist_name_by_module = {module: name for name, module in dist_map.items()}


def _regularize_cov(cov, eps=1e-6):
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = np.min(eigvals)
        if min_eig < eps:
            cov = cov + (eps - min_eig) * np.eye(cov.shape[0])
    except np.linalg.LinAlgError:
        cov = cov + eps * np.eye(cov.shape[0])
    return cov


def _weighted_mvn_update(X, weights, alpha_prev):
    weights = np.asarray(weights, dtype=float)
    n_eff = np.sum(weights)
    if n_eff <= 1e-12:
        return alpha_prev

    mu = np.sum(weights[:, None] * X, axis=0) / n_eff
    diff = X - mu
    cov = (diff.T @ (weights[:, None] * diff)) / n_eff
    return mu, _regularize_cov(cov)


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[0])


def _gh_alpha_bar_from_chi_psi(chi, psi):
    return float(np.sqrt(max(_as_scalar(chi) * _as_scalar(psi), 0.0)))


def _soft_update_gh_family(dist_name, X, weights, alpha_prev, gh_mstep_kwargs):
    if dist_name == "mghp":
        lmbda, chi, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=_as_scalar(lmbda),
            alpha_bar=_gh_alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": True, "alpha_bar": True,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        if fit["alpha_bar"] != 0:
            chi_new, psi_new = mghp._alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
        elif fit["lmbda"] > 0:
            chi_new, psi_new = 0, 2 * fit["lmbda"]
        else:
            chi_new, psi_new = -2 * (fit["lmbda"] + 1), 0
        return fit["lmbda"], chi_new, psi_new, fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mgst":
        lmbda, chi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=_as_scalar(lmbda),
            alpha_bar=0,
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": True, "alpha_bar": False,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        return fit["lmbda"], -2 * (fit["lmbda"] + 1), fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mvhb":
        chi, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=(X.shape[1] + 1) / 2.0,
            alpha_bar=_gh_alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": False, "alpha_bar": True,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        chi_new, psi_new = mghp._alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
        return chi_new, psi_new, fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mnig":
        chi, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=-0.5,
            alpha_bar=_gh_alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": False, "alpha_bar": True,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        return fit["alpha_bar"], fit["alpha_bar"], fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mvvg":
        lmbda, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=_as_scalar(lmbda),
            alpha_bar=0,
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": True, "alpha_bar": False,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        return fit["lmbda"], 2 * fit["lmbda"], fit["mu"], fit["sigma"], fit["gamma"]

    raise NotImplementedError(f"GH-family soft M-step is not available for {dist_name}.")


def _soft_update_one_component(dist_name, X, weights, alpha_prev, gh_mstep_kwargs=None):
    gamma = np.asarray(weights, dtype=float).reshape(-1, 1)
    if np.sum(gamma) <= 1e-12:
        return alpha_prev

    gh_mstep_kwargs = {} if gh_mstep_kwargs is None else gh_mstep_kwargs
    pi_prev = np.array([1.0])
    alpha_list = [alpha_prev]
    p = X.shape[1]

    if dist_name in {"mghp", "mgst", "mvhb", "mnig", "mvvg"}:
        return _soft_update_gh_family(
            dist_name, X, gamma[:, 0], alpha_prev, gh_mstep_kwargs
        )

    if dist_name == "mvn":
        return _weighted_mvn_update(X, gamma[:, 0], alpha_prev)

    if dist_name == "mvsn":
        return estimate_alphas_skewnormal(
            X, gamma, alpha_list, pi_prev,
            dmvSN_func=dmvSN,
            d_mixedmvSN_func=d_mixedmvSN,
        )[0]

    if dist_name == "mvst":
        mu, sigma, lmbda, nu = alpha_prev
        alpha_st = [(mu, sigma, lmbda, _as_scalar(nu))]
        mu_new, sigma_new, lmbda_new, nu_new = estimate_alphas_skewt(
            X, gamma, alpha_st, pi_prev
        )[0]
        return mu_new, sigma_new, lmbda_new, np.array([nu_new])

    if dist_name == "msnc":
        return estimate_alphas_skewcn(
            X, gamma, alpha_list, pi_prev,
            dmvSNC_func=dmvSNC,
            d_mixedmvSNC_func=d_mixedmvSNC,
        )[0]

    if dist_name == "mssl":
        mu, sigma, lmbda, nu = alpha_prev
        alpha_list = [(mu, sigma, lmbda, _as_scalar(nu))]
        mu_new, sigma_new, lmbda_new, nu_new = estimate_alphas_skewslash(
            X, gamma, alpha_list, pi_prev,
            dmvSS_func=dmvSS_skewslash,
            d_mixedmvSS_func=d_mixedmvSS_skewslash,
        )[0]
        return mu_new, sigma_new, lmbda_new, np.array([nu_new])

    if dist_name == "mvsl":
        return estimate_alphas_skewlaplace(X, gamma, alpha_list)[0]

    if dist_name == "mvt":
        loc, shape, df = alpha_prev
        alpha_t = [(loc, shape, np.zeros(p), _as_scalar(df))]
        mu_new, sigma_new, _, nu_new = estimate_alphas_t(
            X, gamma, alpha_t, pi_prev,
            dmvt_ls_func=dmvt_ls,
            d_mixedmvST_func=d_mixedmvST,
        )[0]
        return mu_new, sigma_new, np.array([nu_new])

    if dist_name == "msl":
        mu, sigma, nu = alpha_prev
        alpha_slash = [(mu, sigma, np.zeros(p), _as_scalar(nu))]
        mu_new, sigma_new, _, nu_new = estimate_alphas_slash(
            X, gamma, alpha_slash, pi_prev,
            dmvSS_func=dmvSS_slash,
            d_mixedmvSS_func=d_mixedmvSS_slash,
        )[0]
        return mu_new, sigma_new, np.array([nu_new])

    raise NotImplementedError(f"Soft M-step is not available for {dist_name}.")


def fmm_estimate_alphas_soft(X, gamma_matrix, alpha_prev, soft_dist_comb=None, **kwargs):
    if soft_dist_comb is None:
        raise ValueError("soft_dist_comb must be provided for non-identical soft EM.")
    gh_mstep_kwargs = kwargs.get("gh_mstep_kwargs", {})
    alpha_new = []
    for j, dist_module in enumerate(soft_dist_comb):
        dist_name = dist_name_by_module.get(dist_module)
        if dist_name not in soft_dist:
            raise NotImplementedError(f"Soft M-step is not available for {dist_name}.")
        try:
            alpha_new.append(
                _soft_update_one_component(
                    dist_name, X, gamma_matrix[:, j], alpha_prev[j],
                    gh_mstep_kwargs=gh_mstep_kwargs
                )
            )
        except Exception:
            alpha_new.append(alpha_prev[j])
    return alpha_new

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


def pi_jacobian_from_eta(pi, eps=1e-12):
    """
    Jacobian d pi / d eta for eta_j = log(pi_j / pi_k), j=1,...,k-1.
    """
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    pi = np.clip(pi, eps, 1 - eps)
    pi = pi / pi.sum()

    J = np.zeros((k, k - 1))
    for m in range(k - 1):
        for j in range(k - 1):
            J[j, m] = pi[j] * ((1.0 if j == m else 0.0) - pi[m])
        J[k - 1, m] = -pi[k - 1] * pi[m]
    return J


def eta_score_matrix(tau, pi):
    """
    Per-observation scores for mixture logits eta_j = log(pi_j/pi_k).
    """
    tau = np.asarray(tau, dtype=float)
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    if tau.shape[1] != k:
        raise ValueError("tau and pi have incompatible numbers of components.")
    if k == 1:
        return np.zeros((tau.shape[0], 0))
    return tau[:, :k - 1] - pi[:k - 1]


def safe_component_score_matrix(dist_module, X, params):
    """
    Component score matrix with numerical sanitation.

    The returned scores are on the parameterization used by each distribution's
    score_mat implementation, typically unconstrained Cholesky/log parameters
    for GH/MVN and the package's existing SMSN score coordinates.
    """
    if not hasattr(dist_module, "score_mat") or not callable(getattr(dist_module, "score_mat")):
        raise NotImplementedError(
            f"{dist_module.__name__} does not provide score_mat; cannot compute FMVMM information."
        )

    S = dist_module.score_mat(X, *params)
    S = np.real_if_close(S, tol=1000)
    S = np.asarray(np.real(S), dtype=float)
    if S.ndim != 2 or S.shape[0] != X.shape[0]:
        raise ValueError(
            f"{dist_module.__name__}.score_mat must return shape (n,d); got {S.shape}."
        )

    return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


def observed_score_matrix_fmvmm(X, pi, alpha, dist_comb, tau):
    """
    Full observed-data score matrix for a non-identical finite mixture.

    Parameter order:
      eta_1,...,eta_{k-1}, theta_1, ..., theta_k
    where eta_j = log(pi_j/pi_k), and theta_j follows the score_mat
    parameterization of component j.
    """
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n = X.shape[0]
    k = len(dist_comb)

    if len(alpha) != k or pi.size != k or tau.shape != (n, k):
        raise ValueError("Incompatible X, pi, alpha, dist_comb, and tau dimensions.")

    eta_scores = eta_score_matrix(tau, pi)
    component_scores = []
    raw_component_scores = []
    param_dims = []

    for j, dist_module in enumerate(dist_comb):
        S_j = safe_component_score_matrix(dist_module, X, alpha[j])
        raw_component_scores.append(S_j)
        component_scores.append(tau[:, j:j + 1] * S_j)
        param_dims.append(S_j.shape[1])

    if component_scores:
        S_obs = np.hstack([eta_scores] + component_scores)
    else:
        S_obs = eta_scores

    return S_obs, raw_component_scores, param_dims


def louis_score_decomposition_fmvmm(X, pi, alpha, dist_comb, tau):
    """
    Louis decomposition using complete-data score outer products.

    This computes, observation by observation,
      I_observed(OPG) = E[S_c S_c' | x] - Var[S_c | x]
                      = S_obs S_obs'
    and returns all three summed matrices. It includes the membership-variance
    cross-blocks that the old block-diagonal FMVMM implementation missed.
    """
    S_obs, raw_component_scores, param_dims = observed_score_matrix_fmvmm(
        X, pi, alpha, dist_comb, tau
    )
    tau = np.asarray(tau, dtype=float)
    pi = np.asarray(pi, dtype=float)
    n, k = tau.shape
    eta_dim = max(k - 1, 0)
    total_dim = eta_dim + sum(param_dims)

    offsets = []
    cursor = eta_dim
    for d in param_dims:
        offsets.append(cursor)
        cursor += d

    complete_opg = np.zeros((total_dim, total_dim), dtype=float)

    for i in range(n):
        for h in range(k):
            sc = np.zeros(total_dim, dtype=float)
            if eta_dim:
                sc[:eta_dim] = -pi[:eta_dim]
                if h < eta_dim:
                    sc[h] += 1.0
            start = offsets[h]
            end = start + param_dims[h]
            sc[start:end] = raw_component_scores[h][i]
            complete_opg += tau[i, h] * np.outer(sc, sc)

    observed_opg = S_obs.T @ S_obs
    observed_opg = 0.5 * (observed_opg + observed_opg.T)
    complete_opg = 0.5 * (complete_opg + complete_opg.T)
    missing = complete_opg - observed_opg
    missing = 0.5 * (missing + missing.T)

    return observed_opg, complete_opg, missing, S_obs, param_dims


def classification_information_fmvmm(X, pi, alpha, dist_comb, labels):
    """
    Modified-Louis/classification information for hard EM.

    This treats component labels as known. The missing-information term from
    uncertain membership is therefore zero. The eta block is the multinomial
    complete-data information, and component blocks are OPG component
    information computed only on hard-assigned observations.

    Parameter order is the same internal identifiable order used by the soft
    method:
      eta_1,...,eta_{k-1}, theta_1, ..., theta_k
    """
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = X.shape[0]
    k = len(dist_comb)
    eta_dim = max(k - 1, 0)

    if labels.shape[0] != n:
        raise ValueError("labels must have one entry per observation.")

    component_scores = []
    param_dims = []
    for j, dist_module in enumerate(dist_comb):
        S_all = safe_component_score_matrix(dist_module, X, alpha[j])
        S_j = S_all[labels == j]
        component_scores.append(S_j)
        param_dims.append(S_all.shape[1])

    total_dim = eta_dim + sum(param_dims)
    I = np.zeros((total_dim, total_dim), dtype=float)

    if eta_dim:
        I_eta, _, _ = pi_info_constrained(pi, n)
        I[:eta_dim, :eta_dim] = I_eta

    cursor = eta_dim
    for S_j, d_j in zip(component_scores, param_dims):
        if S_j.shape[0] > 0:
            block = S_j.T @ S_j
            I[cursor:cursor + d_j, cursor:cursor + d_j] = 0.5 * (block + block.T)
        cursor += d_j

    tau_hard = np.zeros((n, k), dtype=float)
    tau_hard[np.arange(n), labels] = 1.0
    S_complete, _, _ = observed_score_matrix_fmvmm(X, pi, alpha, dist_comb, tau_hard)
    missing = np.zeros_like(I)

    return I, I.copy(), missing, S_complete, param_dims


def invert_information_matrix(info, ridge=1e-8, use_pinv=True):
    info = np.asarray(info, dtype=float)
    info = 0.5 * (info + info.T)
    if ridge and ridge > 0:
        info = info + ridge * np.eye(info.shape[0])
    try:
        cov = np.linalg.pinv(info) if use_pinv else np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return info, cov, se


def pi_cov_from_eta_cov(pi, cov_eta):
    if len(pi) <= 1:
        return np.zeros((len(pi), len(pi)))
    J = pi_jacobian_from_eta(pi)
    return J @ cov_eta @ J.T


def user_scale_transform(pi, param_dims):
    """
    Linear delta-method map from internal [eta, theta] to user [pi, theta].
    """
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    eta_dim = max(k - 1, 0)
    theta_dim = int(np.sum(param_dims))
    T = np.zeros((k + theta_dim, eta_dim + theta_dim), dtype=float)
    if eta_dim:
        T[:k, :eta_dim] = pi_jacobian_from_eta(pi)
    if theta_dim:
        T[k:, eta_dim:] = np.eye(theta_dim)
    return T


def transform_information_to_user_scale(pi, param_dims, cov_internal,
                                        ridge=1e-8, use_pinv=True):
    """
    Return covariance/SE for [all pi, theta] by delta method.

    The all-pi covariance is singular because sum(pi)=1. We therefore report a
    generalized information matrix as the Moore-Penrose inverse of the user-scale
    covariance. The SEs are still the natural marginal SEs for all pi values.
    """
    T = user_scale_transform(pi, param_dims)
    cov_user = T @ cov_internal @ T.T
    cov_user = 0.5 * (cov_user + cov_user.T)
    info_user = np.linalg.pinv(cov_user)
    info_user = 0.5 * (info_user + info_user.T)
    se_user = np.sqrt(np.maximum(np.diag(cov_user), 0.0))
    return info_user, cov_user, se_user, T


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
    def __init__(self,n_clusters,tol=0.0001, list_of_dist=all_dist,specific_comb=False,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True, debug = False, em_type="soft", gh_mstep_kwargs=None, init_fit_kwargs=None):
        em_type_normalized = em_type.lower()
        if em_type_normalized not in ("hard", "soft"):
            raise ValueError("em_type must be either 'hard' or 'soft'.")
        EM_type = "Soft" if em_type_normalized == "soft" else "Hard"
        super().__init__(n_clusters=n_clusters, EM_type=EM_type, mixture_type="nonidentical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
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
        self.em_type = em_type_normalized
        self.gh_mstep_kwargs = {} if gh_mstep_kwargs is None else gh_mstep_kwargs
        self.init_fit_kwargs = {} if init_fit_kwargs is None else init_fit_kwargs
        self.init_fit_kwargs_by_module = {
            dist_map[name]: kwargs
            for name, kwargs in self.init_fit_kwargs.items()
            if name in dist_map
        }

    def _log_pdf_non_identical(self,X,alphas,dist_comb):
        N,p=X.shape
        k=len(alphas)
        probs=np.empty((N, k))
        for j in range(k):
            alpha=alphas[j]
            log_prob = np.asarray(dist_comb[j].logpdf(X,*alpha), dtype=float)
            probs[:, j]=np.where(np.isfinite(log_prob), log_prob, -1e300)
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
                        self.data, self.k, self.dist_combs[l],
                        fit_kwargs_by_module=self.init_fit_kwargs_by_module)
                    self.alpha_temp = self.alpha_not
                    self.pi_temp = self.pi_not
                else:
                    self.pi_not, self.alpha_not = fmm_gmm_init(
                        self.data, self.k, self.dist_combs[l],
                        fit_kwargs_by_module=self.init_fit_kwargs_by_module)
                    self.alpha_temp = self.alpha_not
                    self.pi_temp = self.pi_not
                if self.EM_type == "Soft":
                    pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(
                        self.data,
                        self.pi_temp,
                        self.alpha_temp,
                        fmm_estimate_alphas_soft,
                        dist_comb=self.dist_combs[l],
                        soft_dist_comb=self.dist_combs[l],
                        gh_mstep_kwargs=self.gh_mstep_kwargs
                    )
                else:
                    pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(
                        self.data,
                        self.pi_temp,
                        self.alpha_temp,
                        fmm_estimate_alphas,
                        dist_comb=self.dist_combs[l],
                        fit_kwargs_by_module=self.init_fit_kwargs_by_module
                    )
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
        return self.list_pi[np.argmin(self.list_bic)], self.list_alpha[np.argmin(self.list_bic)]

    def best_predict(self):
        return self.list_cluster[np.argmin(self.list_bic)]

    def best_predict_new(self, x):
        data_lol = x.values.tolist()
        cluster_all = []
        for l in range(len(self.dist_combs)):
            cluster, _ = mixture_clusters(self.gamma_matrix[l], data_lol)
            cluster_all.append(cluster)
        return cluster_all[np.argmin(self.list_bic)]

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

    def get_info_mat(self, method="auto", ridge=1e-8, use_pinv=True,
                     parameterization="user", return_details=False):
        """
        Observed information and standard errors for each fitted FMVMM candidate.

        Parameters
        ----------
        method : {"auto", "soft", "score", "opg", "louis",
                  "hard", "classification", "modified_louis"}
            "auto" uses "soft"/"louis" for soft-EM fits and
            "classification" for hard-EM fits.

            Soft methods use the full observed-data score/Louis information,
            including membership uncertainty. Hard/classification methods treat
            labels as known; the missing-information term is zero by
            construction.
        ridge : float
            Ridge added before inversion for numerical stability.
        use_pinv : bool
            Use Moore-Penrose inverse by default; mixture information can be
            nearly singular when components overlap or labels are weakly
            identified.
        return_details : bool
            If True, also return diagnostics and covariance matrices.
        parameterization : {"user", "eta", "internal"}
            "user" returns all pi values followed by component parameters. The
            all-pi covariance is singular because sum(pi)=1, so the returned
            information matrix is a generalized information matrix. "eta" or
            "internal" returns the identifiable eta-logit parameterization.

        Notes
        -----
        Internal parameter order is:
          eta_1,...,eta_{k-1}, theta_1, ..., theta_k
        where eta_j = log(pi_j / pi_k). The returned SE vector is on this
        identifiable scale when parameterization="eta". The default
        parameterization="user" returns SEs for all pi values via delta method.
        """
        method = method.lower()
        parameterization = parameterization.lower()

        if method == "auto":
            method = "classification" if self.EM_type == "Hard" else "louis"
        if method in {"opg", "soft"}:
            method = "score"
        if method in {"hard", "modified_louis", "classification_louis"}:
            method = "classification"
        if method not in {"score", "louis", "classification"}:
            raise NotImplementedError(
                "method must be 'auto', 'score', 'opg', 'louis', "
                "'hard', 'classification', or 'modified_louis'."
            )
        if parameterization not in {"user", "eta", "internal"}:
            raise ValueError("parameterization must be 'user', 'eta', or 'internal'.")

        mixture_info_mats = []
        mixture_ses = []
        details_all = []

        X = np.asarray(self.data, dtype=float)

        for a, mix in enumerate(self.worked_dist):
            pi = np.asarray(self.list_pi[a], dtype=float)
            alpha = self.list_alpha[a]
            tau = np.asarray(self.list_gamma_matrix[a], dtype=float)
            k = len(mix)

            if method == "classification":
                labels = np.asarray(self.list_cluster[a], dtype=int)
                I_raw, complete_opg, missing_info, S_obs, param_dims = (
                    classification_information_fmvmm(X, pi, alpha, mix, labels)
                )
                information_label = "hard_classification_modified_louis"
            elif method == "louis":
                I_raw, complete_opg, missing_info, S_obs, param_dims = (
                    louis_score_decomposition_fmvmm(X, pi, alpha, mix, tau)
                )
                information_label = "soft_louis_score_decomposition"
            else:
                S_obs, _, param_dims = observed_score_matrix_fmvmm(
                    X, pi, alpha, mix, tau
                )
                I_raw = S_obs.T @ S_obs
                I_raw = 0.5 * (I_raw + I_raw.T)
                complete_opg = None
                missing_info = None
                information_label = "soft_observed_score_opg"

            I_internal, cov_internal, se_internal = invert_information_matrix(
                I_raw, ridge=ridge, use_pinv=use_pinv
            )

            eta_dim = max(k - 1, 0)
            cov_eta = cov_internal[:eta_dim, :eta_dim] if eta_dim else np.zeros((0, 0))
            cov_pi = pi_cov_from_eta_cov(pi, cov_eta)
            se_pi = np.sqrt(np.maximum(np.diag(cov_pi), 0.0))
            I_user, cov_user, se_user, user_transform = transform_information_to_user_scale(
                pi, param_dims, cov_internal, ridge=ridge, use_pinv=use_pinv
            )

            score_sum = S_obs.sum(axis=0)
            try:
                condition_number = np.linalg.cond(I_internal)
            except np.linalg.LinAlgError:
                condition_number = np.inf

            if parameterization == "user":
                mixture_info_mats.append(I_user)
                mixture_ses.append(se_user)
                returned_parameterization = "user_all_pi"
            else:
                mixture_info_mats.append(I_internal)
                mixture_ses.append(se_internal)
                returned_parameterization = "eta_internal"

            details_all.append({
                "method": method,
                "information_label": information_label,
                "returned_parameterization": returned_parameterization,
                "internal_parameterization": "eta logits followed by component score_mat blocks",
                "user_parameterization": "all pi followed by component score_mat blocks",
                "eta_dim": eta_dim,
                "component_param_dims": param_dims,
                "info_internal": I_internal,
                "cov_internal": cov_internal,
                "se_internal": se_internal,
                "info_user": I_user,
                "cov_user": cov_user,
                "se_user": se_user,
                "user_transform": user_transform,
                "cov": cov_user if parameterization == "user" else cov_internal,
                "cov_eta": cov_eta,
                "cov_pi": cov_pi,
                "se_pi": se_pi,
                "observed_score_matrix": S_obs,
                "score_sum_norm": float(np.linalg.norm(score_sum)),
                "condition_number": float(condition_number),
                "complete_score_opg": complete_opg,
                "missing_information": missing_info,
            })

        if return_details:
            return mixture_info_mats, mixture_ses, details_all
        return mixture_info_mats, mixture_ses

    def get_info_mat_soft(self, *args, **kwargs):
        """
        Corrected full soft-mixture observed information. See get_info_mat().
        """
        kwargs.setdefault("method", "louis")
        return self.get_info_mat(*args, **kwargs)

    def get_info_mat_hard(self, *args, **kwargs):
        """
        Hard/classification modified-Louis information treating labels as known.
        The missing-information term for membership uncertainty is zero.
        """
        kwargs.setdefault("method", "classification")
        return self.get_info_mat(*args, **kwargs)

    def get_top_mixtures(self,n_top=10):
        wrk_lst=[]
        for i in range(len(self.worked_dist)):
            wrk_lst.append([str(self.worked_dist[i][j].__name__) for j in range(len(self.worked_dist[i]))])

        # Combine the two lists into a list of tuples
        combined = list(zip(self.list_bic, wrk_lst))

        # Sort the list of tuples based on bic values
        sorted_combined = sorted(combined)

        return sorted_combined[:n_top]
