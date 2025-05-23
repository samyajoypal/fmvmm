import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture
from scipy.stats import dirichlet
import math
import pandas as pd
import conorm
from scipy.special import gammaln, psi,digamma
import scipy.special as sp
from scipy.stats import norm

def wald_confidence_intervals_dmm(mle, info_matrix, alpha=0.05):
    """
    Compute Wald confidence intervals for a Dirichlet Mixture Model (DMM).

    Parameters
    ----------
    mle : tuple
        (pi, alphas):
        - pi: list of K mixture weights (summing to 1)
        - alphas: list of K lists of Dirichlet parameters

    info_matrix : ndarray
        The full observed information matrix (negative Hessian of log-likelihood),
        shape: (K + K*D, K + K*D)

    alpha : float
        Significance level for (1 - alpha)*100% confidence intervals

    Returns
    -------
    list of tuples
        Each tuple contains (lower_bound, upper_bound) for each parameter,
        in the order: [pi_1, ..., pi_K, alpha_11, ..., alpha_KD]
    """
    pi, alphas = mle
    K = len(pi)
    D = len(alphas[0])
    total_params = K + K * D

    # Invert the information matrix to get covariance matrix
    cov_matrix = np.linalg.inv(info_matrix)

    z = norm.ppf(1 - alpha / 2)
    ci_list = []

    # --- Step 1: ALR transform of pi and delta method SEs ---
    pi = np.array(pi)
    eta = np.log(pi[:-1] / pi[-1])  # K−1 transformed weights
    for k in range(K - 1):
        # gradient of eta_k = log(pi_k / pi_K)
        grad = np.zeros(K)
        grad[k] = 1 / pi[k]
        grad[K - 1] = -1 / pi[K - 1]
        var_eta_k = grad @ cov_matrix[:K, :K] @ grad.T
        se_eta_k = np.sqrt(var_eta_k)

        # Wald CI in eta space
        eta_low = eta[k] - z * se_eta_k
        eta_high = eta[k] + z * se_eta_k

        # Back-transform to get pi_k bounds
        exp_low = np.exp(eta_low)
        exp_high = np.exp(eta_high)
        sum_exp_low = exp_low + sum(np.exp(np.delete(eta, k)))
        sum_exp_high = exp_high + sum(np.exp(np.delete(eta, k)))
        pi_k_low = exp_low / (1 + sum_exp_low)
        pi_k_high = exp_high / (1 + sum_exp_high)
        ci_list.append((pi_k_low, pi_k_high))

    # Now get CI for pi_K
    grad = np.zeros(K)
    grad[K - 1] = 1 / pi[K - 1]
    var_log_piK = grad @ cov_matrix[:K, :K] @ grad.T
    se_log_piK = np.sqrt(var_log_piK)

    eta_K = np.log(pi[K - 1])
    eta_low = eta_K - z * se_log_piK
    eta_high = eta_K + z * se_log_piK
    ci_list.append((np.exp(eta_low), np.exp(eta_high)))  # CI for pi_K

    # --- Step 2: log-transform alpha and delta method SEs ---
    flat_alphas = [a for alpha_k in alphas for a in alpha_k]
    for i in range(K * D):
        alpha_hat = flat_alphas[i]
        idx = K + i  # offset in full param vector
        var_alpha = cov_matrix[idx, idx]
        se_log_alpha = np.sqrt(var_alpha) / alpha_hat  # delta method
        log_alpha = np.log(alpha_hat)
        ci_low = np.exp(log_alpha - z * se_log_alpha)
        ci_high = np.exp(log_alpha + z * se_log_alpha)
        ci_list.append((ci_low, ci_high))

    return ci_list


def score_vector_observation(pi, alpha, x_i, gamma_i):
    """
    Computes the score vector for a single observation x_i, given:
        - pi: shape (k,)
        - alpha: shape (k, d)
        - x_i: shape (d,)
        - gamma_i: shape (k,)
    Returns:
        - score_i: shape (k + k*d,)
    """
    k, d = alpha.shape
    score_pi = gamma_i / pi  # shape (k,)
    score_alpha = []

    for j in range(k):
        alpha_j = alpha[j]
        alpha_sum = np.sum(alpha_j)
        grad = sp.psi(alpha_sum) - sp.psi(alpha_j) + np.log(x_i)  # shape (d,)
        score_alpha_j = gamma_i[j] * grad
        score_alpha.append(score_alpha_j)

    score_alpha = np.concatenate(score_alpha)  # shape (k*d,)
    return np.concatenate([score_pi, score_alpha])  # shape (k + k*d,)

def empirical_info_matrix(pi, alpha, gamma, x, mode='soft'):
    """
    Computes the empirical observed information matrix using the score vector method.
    mode: 'soft' or 'hard'
    Returns:
        - I_emp: empirical observed info matrix (k + k*d, k + k*d)
    """
    if mode == 'hard':
        gamma = hard_assignments(gamma)

    N, d = x.shape
    k = len(pi)
    big_dim = k + k * d
    I_emp = np.zeros((big_dim, big_dim))

    for i in range(N):
        score_i = score_vector_observation(pi, alpha, x[i], gamma[i])
        I_emp += np.outer(score_i, score_i)

    return I_emp

def missing_info_pi(pi, gamma):
    """
    Computes the missing information matrix (covariance of score) for π.

    Parameters:
    - pi: shape (K,)
    - gamma: shape (n, K)

    Returns:
    - I_missing: shape (K, K)
    """
    n, K = gamma.shape
    I_missing = np.zeros((K, K))

    for k in range(K):
        for j in range(K):
            if k == j:
                I_missing[k, k] = np.sum(gamma[:, k] * (1 - gamma[:, k])) / pi[k]**2
            else:
                I_missing[k, j] = -np.sum(gamma[:, k] * gamma[:, j]) / (pi[k] * pi[j])

    return I_missing

def missing_info_alpha(alpha_j, gamma_j, x):
    """
    Compute the missing information matrix for one Dirichlet component's alpha_j.
    gamma_j: (N,) vector of posterior probs for component j
    x: (N, d) data matrix
    """
    N, d = x.shape
    alpha_sum = np.sum(alpha_j)
    psi_sum = sp.psi(alpha_sum)
    psi_j = sp.psi(alpha_j)

    A = psi_sum - psi_j + np.log(x)  # shape (N, d)

    # Compute missing info matrix: sum_i gamma_{ij} (1 - gamma_{ij}) A_i A_i^T
    weights = gamma_j * (1 - gamma_j)  # shape (N,)
    I_miss = np.zeros((d, d))
    for i in range(N):
        outer = np.outer(A[i], A[i])
        I_miss += weights[i] * outer
    return I_miss


def hard_assignments(gamma):
    """
    Convert soft responsibilities (rows sum to 1) into hard 0/1 assignments
    by taking argmax in each row. Returns a new array with exactly one '1'
    per row, and zeros elsewhere.
    """
    n_data, k = gamma.shape
    hard = np.zeros_like(gamma)
    max_indices = np.argmax(gamma, axis=1)
    hard[np.arange(n_data), max_indices] = 1.0
    return hard

def mixture_counts(gamma, mode='soft'):
    """
    Depending on mode='soft' or 'hard', return the cluster counts N_j for j=1..k
    and the total N = sum_j N_j.
    """
    if mode == 'hard':
        gamma_hard = hard_assignments(gamma)
        N_j = np.sum(gamma_hard, axis=0)  # integer counts
    else:
        N_j = np.sum(gamma, axis=0)      # sum of partial responsibilities
    N = np.sum(N_j)
    return N_j, N

def mixture_proportions_info(pi, N_j):
    """
    For mixture weights pi (length k), compute the unconstrained (k x k)
    information matrix I_pi and its inverse, using:
       I_pi       = diag(N_j / pi_j^2),
       I_pi_inv   = diag(pi_j^2 / N_j).
    """
    # pi, N_j should each be shape (k,)
    I_diag = N_j / (pi**2)
    I_pi = np.diag(I_diag)

    I_inv_diag = (pi**2) / N_j
    I_pi_inv = np.diag(I_inv_diag)

    return I_pi, I_pi_inv

def single_dirichlet_info(alpha_j, n_j, ridge_factor=1e-10, use_pinv=False):
    """
    Compute the (d x d) Fisher info matrix and its inverse for ONE Dirichlet
    parameter vector alpha_j of length d, given 'n_j' as the effective sample size
    for that cluster (either sum of responsibilities if soft or integer if hard).

    Using the closed-form:
        I_alpha = D + G * 11^T,
        I_alpha^-1 = D_star + beta * (a_star)(a_star)^T,

    where:
      - D = diag(n_j * psi'(alpha_j)),
      - G = - n_j * psi'(sum(alpha_j)),
      - D_star = diag(1 / [n_j * psi'(alpha_j)] ),
      - a_star = [1 / psi'(alpha_j1), ..., 1 / psi'(alpha_jd]]^T,
      - beta = [n_j * psi'(sum alpha_j)] / [1 - psi'(sum alpha_j)* sum(1/psi'(alpha_j))].
    """
    d = len(alpha_j)
    alpha_sum = np.sum(alpha_j)

    # D = diag(n_j * trigamma(alpha_j))
    D_vals = n_j * sp.polygamma(1, alpha_j)  # shape (d,)
    D = np.diag(D_vals)

    # G = - n_j * trigamma(sum_j alpha_j)
    psi_sum = sp.polygamma(1, alpha_sum)
    G = - n_j * psi_sum

    # => I_alpha (d x d)
    I_alpha = D + G * np.ones((d, d))

    # # Inverse
    # D_star = np.diag(1.0 / D_vals)  # shape (d x d)
    # a_star_vals = 1.0 / sp.polygamma(1, alpha_j)  # shape (d,)
    # a_star = a_star_vals.reshape(-1,1)            # (d,1)

    # denom = 1.0 - psi_sum * np.sum(a_star_vals)
    # if abs(denom) < 1e-15:
    #     raise ValueError("Denominator for beta is near zero; check alpha_j range or n_j.")

    # beta = (n_j * psi_sum) / denom
    # I_alpha_inv = D_star + beta * (a_star @ a_star.T)

    # I_alpha_inv = np.linalg.inv(I_alpha)
    M = I_alpha.shape[0]
    # First attempt direct inverse
    try:
        I_alpha_inv = np.linalg.inv(I_alpha)
    except np.linalg.LinAlgError:
        # Direct inverse failed
        if use_pinv:
            # Use the pseudoinverse
            I_alpha_inv = np.linalg.pinv(I_alpha)
        else:
            # Try adding a small ridge on the diagonal
            ridge_mat = I_alpha + ridge_factor*np.eye(M)
            # In case that also fails, we do a nested try:
            try:
                I_alpha_inv = np.linalg.inv(ridge_mat)
            except np.linalg.LinAlgError:
                # As a last fallback, use pseudoinverse of the ridge version
                I_alpha_inv = np.linalg.pinv(ridge_mat)

    return I_alpha, I_alpha_inv

def combined_info_and_se(pi, alpha, gamma, x,method ='score', mode='soft'):
    """
    Computes observed Fisher Information and SEs by subtracting missing info from complete info.
    Returns:
        I_obs: observed info matrix
        se_obs: sqrt of diagonal of inv(I_obs)
    """
    if method =="louis":
        N_j, N = mixture_counts(gamma, mode=mode)
        k = len(pi)
        d = alpha.shape[1]
        big_dim = k + k * d

        # Complete info
        I_pi, I_pi_inv = mixture_proportions_info(pi, N_j)
        I_blocks = []
        I_inv_blocks = []
        for j in range(k):
            I_j, I_j_inv = single_dirichlet_info(alpha[j], N_j[j])
            I_blocks.append(I_j)
            I_inv_blocks.append(I_j_inv)

        # Build complete info matrix
        I_comp = np.zeros((big_dim, big_dim))
        I_comp[0:k, 0:k] = I_pi
        for j in range(k):
            row_start = k + j * d
            I_comp[row_start:row_start + d, row_start:row_start + d] = I_blocks[j]

        # Compute missing info
        I_miss = np.zeros_like(I_comp)
        I_miss[0:k, 0:k] = missing_info_pi(pi, gamma)

        for j in range(k):
            row_start = k + j * d
            I_miss_alpha_j = missing_info_alpha(alpha[j], gamma[:, j], x)
            I_miss[row_start:row_start + d, row_start:row_start + d] = I_miss_alpha_j

        # Observed Info
        I_obs = I_comp - I_miss

        # Attempt inverse and standard errors
        try:
            I_obs_inv = np.linalg.inv(I_obs)
        except np.linalg.LinAlgError:
            I_obs_inv = np.linalg.pinv(I_obs)

        var_diag = np.diag(I_obs_inv)
        # print(var_diag)
        se_obs = np.sqrt(var_diag)

        return I_obs, se_obs
    elif method == "score":
        IM = empirical_info_matrix(pi, alpha, gamma, x, mode)
        try:
            IM_inv = np.linalg.inv(IM)
        except np.linalg.LinAlgError:
            IM_inv = np.linalg.pinv(IM)
        SE = np.sqrt(np.diag(IM_inv))

        return IM, SE
    else:
        raise NotImplementedError("Please use louis or score method. Other methods are not implemented!")

def dirichlet_kl_divergence(alpha1, alpha2):
    """
    Compute KL divergence between two Dirichlet distributions.

    Parameters:
        alpha1 (numpy array): Parameters of the first Dirichlet distribution.
        alpha2 (numpy array): Parameters of the second Dirichlet distribution.

    Returns:
        float: KL divergence between the two Dirichlet distributions.
    """
    sum_alpha1 = np.sum(alpha1)
    sum_alpha2 = np.sum(alpha2)

    term1 = gammaln(sum_alpha1) - gammaln(sum_alpha2)
    term2 = np.sum(gammaln(alpha2) - gammaln(alpha1))
    term3 = np.sum((alpha1 - alpha2) * (digamma(alpha1) - digamma(sum_alpha1)))

    kl_divergence = term1 + term2 + term3
    return kl_divergence

def dmm_kl_divergence(pi, omega, f_models, g_models):
    """
    Compute KL divergence between two Dirichlet mixture models.

    Parameters:
        pi (numpy array): Mixing coefficients for the first mixture model.
        omega (numpy array): Mixing coefficients for the second mixture model.
        f_models (list of numpy arrays): Parameters of Dirichlet distributions in the first mixture model.
        g_models (list of numpy arrays): Parameters of Dirichlet distributions in the second mixture model.

    Returns:
        float: KL divergence between the two Dirichlet mixture models.
    """
    kl_divergence = 0.0

    for a in range(len(pi)):
        term_num = 0.0
        term_denom = 0.0

        for aprime in range(len(pi)):
            term_num += pi[aprime] * np.exp(-dirichlet_kl_divergence(f_models[a], f_models[aprime]))

        for b in range(len(omega)):
            term_denom += omega[b] * np.exp(-dirichlet_kl_divergence(f_models[a], g_models[b]))

        if term_denom==0:
            term_denom=term_denom+1e-243
        if term_num==0:
            term_num=term_num+1e-243
        kl_divergence += pi[a] * np.log(term_num / term_denom)

    return kl_divergence


def dmm_mc_kl_divergence(pi, omega, f_models, g_models, n_samples=1000):
    """
    Monte Carlo estimation of KL Divergence between two Dirichlet mixture models.

    Parameters:
        pi (numpy array): Mixing coefficients for the first mixture model.
        omega (numpy array): Mixing coefficients for the second mixture model.
        f_models (2D numpy array): Parameters of Dirichlet distributions in the first mixture model.
        g_models (2D numpy array): Parameters of Dirichlet distributions in the second mixture model.
        n_samples (int): Number of samples to draw.

    Returns:
        float: Estimated KL Divergence.
    """
    np.random.seed(0)
    k=len(pi)
    assert n_samples>k
    nis=[int(n_samples*i) for i in pi]
    xis=[]
    for j in range(k):
        xis.extend(np.random.dirichlet(f_models[j],nis[j]))

    logs = []

    for xi in xis:
        # # Draw a sample xi from the mixture model using pi and f_models
        # xi_=[pi[j] * np.random.dirichlet(f_models[j]) for j in range(len(pi))]
        # xi = np.sum(xi_,axis=0)
        # # print(xi_)
        # Calculate f_pdf and g_pdf
        with np.errstate(under='ignore'):
            f_pdf_=[pi[j] * dirichlet.pdf(xi, f_models[j]) for j in range(len(pi))]
            f_pdf = np.nansum(np.array(f_pdf_,dtype=np.float128))
            g_pdf_=[omega[j] * dirichlet.pdf(xi, g_models[j]) for j in range(len(omega))]
            g_pdf = np.nansum(np.array(g_pdf_,dtype=np.float128))

            # Calculate log(f(x_i)/g(x_i)) and append to logs
            logs.append(np.log(f_pdf / g_pdf))

    # Average over all samples
    return np.nansum(logs)/len(logs)



def sample_dirichlet_gibbs(alpha, n_iterations=1000):
  """
  This function draws samples from a Dirichlet distribution using the Gibbs sampler and averages them.

  Args:
      alpha: A numpy array of length n_components representing the Dirichlet parameters.
      n_samples: The number of samples to draw.

  Returns:
      A numpy array of shape (n_samples, n_components) containing the average of the samples.
  """

  # Initialize an empty list to store samples
  samples = []

  # Iterate for the desired number of samples
  for _ in range(n_iterations):
    # Generate a single sample
    sample = _sample_dirichlet_gibbs(alpha)  # Helper function for one sample
    samples.append(sample)

  # Convert the list of samples to a numpy array and return the average
  return np.mean(np.array(samples), axis=0)


def _sample_dirichlet_gibbs(alpha, n_samples=1):
  """
  This function draws samples from a Dirichlet distribution using the Gibbs sampler.

  Args:
      alpha: A numpy array of length n_components representing the Dirichlet parameters.
      n_samples: The number of samples to draw.

  Returns:
      A numpy array of shape (n_samples, n_components) containing the samples.
  """
  n_components=len(alpha)
  # Initialize the samples matrix with random values from a uniform distribution
  samples = np.random.rand(n_samples, n_components)

  # Normalize each row to sum to 1 (represents probability distribution)
  samples /= samples.sum(axis=1, keepdims=True)

  for _ in range(n_samples - 1):
    # Iterate over each component
    for k in range(n_components):
      # Calculate the sum of alphas excluding the current component
      other_alpha_sum = alpha.sum() - alpha[k]
      # Sample from a beta distribution using the updated alpha values
      samples[:, k] = np.random.beta(alpha[k] + samples[:, :k].sum(axis=1), other_alpha_sum + samples[:, k+1:].sum(axis=1))

  return samples.flatten()


def dmm_mcmc_kl_divergence(pi, omega, f_models, g_models, n_samples=1000):
    """
    Monte Carlo estimation of KL Divergence between two Dirichlet mixture models.

    Parameters:
        pi (numpy array): Mixing coefficients for the first mixture model.
        omega (numpy array): Mixing coefficients for the second mixture model.
        f_models (2D numpy array): Parameters of Dirichlet distributions in the first mixture model.
        g_models (2D numpy array): Parameters of Dirichlet distributions in the second mixture model.
        n_samples (int): Number of samples to draw.

    Returns:
        float: Estimated KL Divergence.
    """
    logs = []
    np.random.seed(0)

    for _ in range(n_samples):
        # Draw a sample xi from the mixture model using pi and f_models
        xi = np.sum([pi[j] * sample_dirichlet_gibbs(f_models[j]) for j in range(len(pi))],axis=0)
        # print(xi)
        # Calculate f_pdf and g_pdf
        f_pdf = np.sum([pi[j] * dirichlet.pdf(xi, f_models[j]) for j in range(len(pi))])
        g_pdf = np.sum([omega[j] * dirichlet.pdf(xi, g_models[j]) for j in range(len(omega))])

        # Calculate log(f(x_i)/g(x_i)) and append to logs
        logs.append(np.log(f_pdf / g_pdf))

    # Average over all samples
    return np.mean(logs)



def closure(d_mat):
    d_mat = np.atleast_2d(d_mat)
    if np.any(d_mat < 0):
        raise ValueError("Cannot have negative proportions")
    if d_mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(d_mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    d_mat = d_mat / d_mat.sum(axis=1, keepdims=True)
    return d_mat.squeeze()

def multiplicative_replacement(d_mat, delta=None):
    d_mat = closure(d_mat)
    z_mat = (d_mat == 0)

    num_feats = d_mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError('The multiplicative replacement created negative '
                         'proportions. Consider using a smaller `delta`.')
    d_mat = np.where(z_mat, delta, zcnts * d_mat)
    return d_mat.squeeze()


def clr(mat):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    """
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()










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
    p = data.shape[1]
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
    data_lol = data.tolist()
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
    data_lol = data.tolist()
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


def count_to_comp(df):
    df_array=np.array(df)


    nf  = conorm.tmm_norm_factors(df)["norm.factors"]

    lj=[]
    for j in range(df_array.shape[1]):
        lj_temp=nf[j]*np.sum(df_array[:, j])
        lj.append(lj_temp)


    sj=[]
    for j in range(df_array.shape[1]):
        sj_temp=lj[j]/(np.sum(lj)/df_array.shape[1])
        sj.append(sj_temp)

    x_lol=[]
    for i in range(df_array.shape[0]):
        xi=[]
        for j in range(df_array.shape[1]):
            xi_temp=df_array[i,j]/sj[j]
            xi.append(xi_temp)
        xi_sum=np.sum(xi)
        xi_trans=[xi[k]/xi_sum for k in range(df_array.shape[1])]


        x_lol.append(xi_trans)
        #x_lol.append(xi)

    data=pd.DataFrame(x_lol)
    # trans_data=pd.DataFrame(multiplicative_replacement(data))
    trans_data=pd.DataFrame(data)

    return trans_data




def dirichlet_covariance(alpha):
    p = len(alpha)
    alpha0 = np.sum(alpha)
    cov = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            if i == j:
                cov[i, j] = (alpha[i] * (alpha0 - alpha[i])) / (alpha0**2 * (alpha0 + 1))
            else:
                cov[i, j] = -(alpha[i] * alpha[j]) / (alpha0**2 * (alpha0 + 1))

    return cov
