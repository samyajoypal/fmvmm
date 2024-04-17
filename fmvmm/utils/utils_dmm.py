import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture
from scipy.stats import dirichlet
import math
import pandas as pd
import conorm
from scipy.special import gammaln, psi,digamma
import dirichlet as drm

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
