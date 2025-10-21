# Standard library: general utilities
import inspect
import os
import pickle
import hashlib
from functools import wraps

# NumPy and Matplotlib: math and plotting
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt

# SciPy: probability distributions, math functions, and optimization
# Hint: these tools might be useful later in the project
from scipy.stats import gamma, norm, wishart, multivariate_normal
from scipy.spatial.transform import Rotation as R
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize
from scipy.special import gammaln


# DIPY: diffusion MRI utilities and models
from dipy.io.image import load_nifti, save_nifti   # for loading / saving imaging datasets
from dipy.io.gradients import read_bvals_bvecs     # for loading / saving our bvals and bvecs
from dipy.core.gradients import gradient_table     # for constructing gradient table from bvals/bvecs
from dipy.data import get_fnames                   # for small datasets that we use in tests and examples
from dipy.segment.mask import median_otsu          # for masking out the background
import dipy.reconst.dti as dti                     # for diffusion tensor model fitting and metrics



"""
=============================================================================
Caching Utility (already implemented)
=============================================================================
Provides disk-based memoization to avoid recomputation.
"""

def disk_memoize(cache_dir="cache"):
    """
    Decorator for caching function outputs on disk.

    This utility is already implemented and should not be modified by students.
    It allows expensive computations to be stored and re-used across runs,
    based on the function arguments. If you call the same function again with
    the same inputs, it returns the cached results instead of recomputing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result
        
        return wrapper
    return decorator



"""
=============================================================================
Data Loading & Preprocessing (already implemented)
=============================================================================
Loads the Stanford HARDI dataset, applies masking/cropping, and
extracts one voxel with a DTI point estimate for testing.
"""

@disk_memoize()
def get_preprocessed_data():
    """
    Load and preprocess a single voxel of diffusion MRI data.

    What it does:
    - Loads the dataset and gradient information (b-values and b-vectors).
    - Fits a diffusion tensor model (DTI) to one voxel.
    - Extracts a point estimate: baseline signal (S0), eigenvalues, eigenvectors.

    Returns
    -------
    y : ndarray
        Observed diffusion MRI signal vector for a single voxel.
    point_estimate : [S0, evals, evecs]
        Estimated baseline signal, eigenvalues, and eigenvectors.
    gtab : GradientTable
        Gradient table with b-values (diffusion weighting strength)
        and b-vectors (gradient directions).
    """

    # Load the masked data, background mask, and gradient information
    data, mask, gtab = get_data()

    # Initialize a diffusion tensor model (DTI) with S0 estimation enabled
    tenmodel = dti.TensorModel(gtab, return_S0_hat=True)

    # Extract the signal for a single voxel (coordinates chosen for this project)
    y = data[35, 35, 30, :]

    # Fit the DTI model to this voxel's signal
    tenfit = tenmodel.fit(y)
    
    # Extract point estimates: baseline signal, eigenvalues, and eigenvectors
    S0 = tenfit.S0_hat
    evals = tenfit.evals
    evecs = tenfit.evecs
    point_estimate = [S0, evals, evecs]

    # Return the raw voxel signal, point estimate, and gradient table
    return y, point_estimate, gtab 



def get_data():
    """
    Load and preprocess the Stanford HARDI diffusion MRI dataset.

    What it does:
    - Downloads the dataset if not already present (via DIPY).
    - Loads the 4D diffusion MRI volume (x, y, z, measurements).
    - Reads b-values (diffusion weighting strength) and b-vectors (gradient directions).
    - Creates a gradient table (gtab) combining this information.
    - Applies a brain mask and cropping to remove background and reduce size.

    Returns
    -------
    maskdata : ndarray
        The masked and cropped diffusion MRI data.
    mask : ndarray (boolean)
        The brain mask used to exclude background voxels.
    gtab : GradientTable
        Gradient information (b-values and b-vectors) for each measurement.
    """

    # Download filenames for the Stanford HARDI dataset if not already cached 
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    # Load the raw 4D dataset: dimensions are (x, y, z, diffusion measurements)
    data, _ = load_nifti(hardi_fname)

    # Read diffusion weighting information (b-values, b-vectors) and build gradient table
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    # Apply brain masking and cropping to remove background and save compute
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
    )

    # Print the final data shape for confirmation
    print('Loaded data with shape: (%d, %d, %d, %d)' % maskdata.shape)

    return maskdata, mask, gtab


"""
=============================================================================
Linear Algebra Helpers (already implemented)
=============================================================================
Functions for reconstructing tensors and switching between
parameterizations. Already implemented.
Hint: you will make use of these helpers later in the project,
the ones involving theta are useful for VI and Laplace.
"""

def compute_D(evals, V):
    """
    Reconstruct the diffusion tensor D from eigenvalues and eigenvectors.

    D = V Λ V.T, where Λ is the diagonal matrix of eigenvalues.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues, shape (3,) or batched.
    V : ndarray
        Eigenvectors, shape (3, 3) or batched.

    Returns
    -------
    D : ndarray
        Diffusion tensor(s), shape (..., 3, 3).
    """

    # Ensure inputs have the correct batch dimensions
    if evals.ndim == 1:
        evals = evals[None, None, :]
    elif evals.ndim == 2:
        evals = evals[:, None, :]
    if V.ndim == 2:
        V = V[None, :, :]

    # Compute D = V Λ V.T as V (V @ Λ).T
    V_scaled = V * evals
    D = np.matmul(V, np.transpose(V_scaled, axes=[0, 2, 1]))

    return D


def theta_from_D(D):
    """
    Convert a diffusion tensor D into an unconstrained parameter vector theta.

    Follows Eq. (18): D = L L.T with L from Cholesky factorization.
    Diagonals are log-transformed, off-diagonals kept raw.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    theta : ndarray (6,)
        Unconstrained parameter vector corresponding to the lower-triangular
        entries of L (log of diagonals, raw off-diagonals).
    """
    
    # Compute Cholesky factor (lower-triangular L) of D
    L = np.linalg.cholesky(D)
    
    # Indices of lower-triangular entries (including diagonal)
    p = D.shape[0]
    tril_indices = np.tril_indices(p)
    theta = []

    # Store log of diagonal entries, raw off-diagonal entries
    for i, j in zip(*tril_indices):
        if i == j:
            theta.append(np.log(L[i, j]))   # Diagonal: log-transform
        else:
            theta.append(L[i, j])           # Off-diagonal: raw value

    return np.array(theta)


def D_from_theta(theta):
    """
    Convert unconstrained parameter vector theta back into diffusion tensor D.

    Follows Eq. (18): D = L L.T with L constructed from theta.
    Diagonal entries of L are exponentiated to ensure positivity,
    off-diagonals are used as raw values.

    Parameters
    ----------
    theta : ndarray (..., 6)
        Unconstrained parameters corresponding to the lower-triangular
        entries of L (log-diagonals, raw off-diagonals).

    Returns
    -------
    D : ndarray (..., 3, 3)
        Symmetric positive-definite diffusion tensor(s).
    """
    
    # Ensure theta is an array and check shape
    theta = np.asarray(theta)
    *batch_shape, _ = theta.shape
    assert theta.shape[-1] == 6, "Last dimension must be 6 for 3x3 lower-triangular matrices."

    # Initialize lower-triangular matrix L
    L = np.zeros((*batch_shape, 3, 3), dtype=theta.dtype)

    # Fill L with exponentiated diagonals and raw off-diagonals
    tril_indices = np.tril_indices(3)
    for k, (i, j) in enumerate(zip(*tril_indices)):
        if i == j:
            L[..., i, j] = np.exp(theta[..., k])   # Diagonal
        else:
            L[..., i, j] = theta[..., k]           # Off-diagonal

    # Reconstruct D = L @ L.T (batch-aware matrix multiplication)
    D = L @ np.swapaxes(L, -1, -2)

    return D.squeeze()


def grad_D_wrt_theta_at_D(D):
    """
    Compute nabla_theta D evaluated at D.

    Uses the parameterization in Eq. (18): D = L L.T where L is built from 
    theta. Returns the gradient tensor with one (3x3) slice per theta component.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    grad_D : ndarray (3, 3, 6)
        Gradient of D w.r.t. theta, one 3x3 matrix per parameter.
    """
    
    # Get Cholesky factor of D and set up indices for lower-triangular entries
    p = D.shape[0]
    L = np.linalg.cholesky(D)
    tril_indices = np.tril_indices(p)
    num_params = len(tril_indices[0])

    # Prepare output container
    grad_D = np.zeros((p, p, num_params))

    # Loop over all parameters in theta
    for k, (m, n) in enumerate(zip(*tril_indices)):

        # Build a basis matrix for the effect of this parameter
        E_mn = np.zeros((p, p))
        if m == n:
            # Diagonal: dL_mm/dtheta = L_mm since L_mm = exp(theta)
            factor = L[m, n]
        else:
            # Off-diagonal: dL_mn/dtheta = 1
            factor = 1.0
        E_mn[m, n] = factor

        # Work out the corresponding change in D
        dD_k = E_mn @ L.T + L @ E_mn.T
        grad_D[:, :, k] = dD_k

    return grad_D


"""
=============================================================================
Bayesian Model Components (need to be implemented)
=============================================================================
Students: implement all parts in this section (priors, likelihoods, etc.)
These are required before any inference method can be attempted.
"""

class frozen_prior:
    # Placeholder for the prior distribution.
    # Hint: you may want to add input parameters to these methods.
 
    def __init__(self, sigma=29, alpha_s=2, theta_s=500,alpha_lam=4, theta_lam=2.5e-4):
        # Parameters
        self.sigma = sigma
        self.alpha_s = alpha_s
        self.theta_s = theta_s
        self.alpha_lam = alpha_lam
        self.theta_lam = theta_lam
        
    def rvs(self, size=None):
        # Size = None makes it a scalar instead of a size 1 vector
        gamma_s = np.random.gamma(self.alpha_s, self.theta_s, size)
        gamma_lam_1 = np.random.gamma(self.alpha_lam, self.theta_lam, size)
        gamma_lam_2 = np.random.gamma(self.alpha_lam, self.theta_lam, size)
        gamma_lam_3 = np.random.gamma(self.alpha_lam, self.theta_lam, size)
        V = R.random()
        return V, gamma_s, gamma_lam_1, gamma_lam_2, gamma_lam_3 
        
    def logpdf(self, gamma_s, gamma_lam_1,gamma_lam_2, gamma_lam_3, V=None):
        """
        x (gamma): The value at which to evaluate the probability density function.

        a (or s, df, c, etc.): This is a distribution-specific shape parameter. For example: 
        In the lognorm distribution, a is the shape parameter. In the gamma distribution, a is the shape parameter. 

        loc: The location parameter, which shifts the distribution's center. For a normal distribution, it corresponds to the mean. 
        
        scale: The scale parameter, which stretches or compresses the distribution. For a normal distribution, it corresponds to the standard deviation. 
        """
        logp_s = gamma.logpdf(gamma_s, a=self.alpha_s, scale=self.theta_s)
        logp_lam_1 = gamma.logpdf(gamma_lam_1, a=self.alpha_lam, scale=self.theta_lam)
        logp_lam_2 = gamma.logpdf(gamma_lam_2, a=self.alpha_lam, scale=self.theta_lam)
        logp_lam_3 = gamma.logpdf(gamma_lam_3, a=self.alpha_lam, scale=self.theta_lam)
        
        # Rotation term (uniform over SO(3), constant, so log prob = 0)
        logp_V = 0.0  
        
        return logp_s + logp_lam_1 + logp_lam_2 + logp_lam_3 + logp_V
    
    
class frozen_likelihood:
    # Placeholder for the likelihood (with partial code provided).
    # Hint: you may want to add input parameters to these methods.

    def __init__(self, gtab,y, point_estimate, variance):
        self.gtab = gtab   # store gradient table with b-values and b-vectors
        self.y = y # added yesterday
        self.point_estimate = point_estimate # added yesterday
        self.variance = variance
       
    def logpdf(self, S0, evecs, evals):
        S0 = np.atleast_1d(S0)        # ensure S0 is array-like
        D = compute_D(evals, evecs)   # reconstruct diffusion tensor

        # Build q from diffusion gradients (b-values & b-vectors),
        # corresponds to the experimental setting x in the project instructions
        q = np.sqrt(self.gtab.bvals[:, None]) * self.gtab.bvecs

        # Model signal S given tensor D and baseline S0
        S = S0[:, None] * np.exp( - np.einsum('...j, ijk, ...k->i...', q, D, q))

        logp = norm.logpdf(self.y, loc=S, scale=self.variance)
        
        return np.sum(logp)

"""
=============================================================================
Posterior Approximations (need to be implemented)
=============================================================================
Students: implement these approximations, which are only used in the
corresponding inference methods below:
  - variational_posterior: used only for Variational Inference
  - mvn_reparameterized: used only for Laplace Approximation

They are NOT needed for Metropolis-Hastings or Importance Sampling.
"""

class variational_posterior:
    # Placeholder for variational posterior approximation.
    # Hint: you may want to add input parameters to these methods.
    # The score() method is already implemented and can be used later
    # when implementing inference (with REINFORCE leave-one-out estimator).

    def __init__(self, D, theta_1, theta_2, theta_9):
        self.theta_3_to_8 = theta_from_D(D)
        self.Sigma = D_from_theta(self.theta_3_to_8)
        self.shape = np.exp(theta_1)
        self.scale = np.exp(theta_2)
        self.df = np.exp(theta_9) + 2

    def set_theta(self, theta):
        self.shape = np.exp(theta[0])
        self.scale = np.exp(theta[1])
        self.Sigma = D_from_theta(theta[2:8])
        self.df = np.exp(theta[8]) + 2

    def logpdf(self, S0_samples, D_samples):
        log_S0 = gamma.logpdf(S0_samples, a=self.shape, scale=self.scale)
        log_D = wishart.logpdf(D_samples, scale=self.Sigma, df=self.df)
        return log_S0 + log_D
    
    def rvs(self, size):
        S0_samples = np.random.gamma(self.shape, self.scale, size=size)
        D_samples = wishart.rvs(scale=self.Sigma, df=self.df, size=size)
        evals_samples, evecs_samples = eigh(D_samples)

        return S0_samples, evals_samples, evecs_samples

    def score(self, S0, D):
        # Combine score contributions from gamma and Wishart parts
        score_wrt_log_shape, score_wrt_log_scale = self.gamma_score(S0)
        score_wrt_theta, score_wrt_log_df = self.wishart_score(D)
        return np.concatenate((
            [score_wrt_log_shape], [score_wrt_log_scale], score_wrt_theta, [score_wrt_log_df])
        )

    def gamma_score(self, x):
        # Score function for gamma distribution
        score_wrt_log_shape = (np.log(x / self.scale) - digamma(self.shape)) * self.shape
        score_wrt_log_scale = (x / self.scale**2 - self.shape / self.scale) * self.scale
        return score_wrt_log_shape, score_wrt_log_scale

    def wishart_score(self, D):
        # Score function for Wishart distribution
        W = self.df * D
        Sigma_inv = np.linalg.inv(self.Sigma)
        score_wrt_Sigma = 0.5 * Sigma_inv @ (W - self.df * self.Sigma) @ Sigma_inv
        score_wrt_theta = np.tensordot(
            score_wrt_Sigma, grad_D_wrt_theta_at_D(self.Sigma), axes=([0,1], [0,1])
        )
        p = W.shape[0]
        _, logdet_W = np.linalg.slogdet(W)
        _, logdet_Sigma = np.linalg.slogdet(self.Sigma)
        digamma_sum = np.sum([digamma((self.df + 1 - j) / 2.0) for j in range(1, p+1)])
        score_wrt_log_df = ((self.df - 2) / 2) * (logdet_W - p * np.log(2) - logdet_Sigma - digamma_sum)
        return score_wrt_theta, score_wrt_log_df


"""
=============================================================================
Inference Methods (need to be implemented)
=============================================================================
Students: implement one method each (MH, IS, VI, or Laplace).
Uses memoization to speed up repeated runs.
"""

@disk_memoize()
def metropolis_hastings():
    raise NotImplementedError


@disk_memoize()
def importance_sampling(n_samples, gamma_param, nu_param):
    # Students: implement Importance Sampling here.
    # Before starting, make sure the prior and likelihood are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    raise NotImplementedError

    return importance_weights, S0_samples, evals_samples, evecs_samples


@disk_memoize()
def variational_inference(max_iters, K, learning_rate):
    # Students: implement Variational Inference here.
    # Before starting, make sure the prior, likelihood and variational_posterior are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    #raise NotImplementedError

    # Initialize data
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=True)
    S0_init, evals_init, evecs_init = point_estimate
    D_ref = compute_D(evals_init, evecs_init).squeeze() # Compute D from initial evals and evecs
    theta_1 = 3
    theta_2 = 3
    theta_9 = 3
    Sigma = D_ref

    # Create array of theta values initialized by D_ref and theta 1, 2 and 9
    theta = np.zeros(9)
    theta[0] = theta_1
    theta[1] = theta_2
    theta_3_to_8 = theta_from_D(Sigma)
    for i in range(len(theta_3_to_8)):
        theta[i+2] = theta_3_to_8[i]
    theta[8] = theta_9

    # Adam parameteres
    m = np.zeros(9)
    v = np.zeros(9)
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    t = 0

    # Create instance of variational posterior, frozen prior and frozen likelihood
    var_posterior = variational_posterior(Sigma, theta_1, theta_2, theta_9) # Initialize variational posterior
    fro_prior = frozen_prior() # Initialize frozen_prior class
    fro_likelihood = frozen_likelihood(gtab, y, point_estimate, variance=29) # Initialize frozen_likelihood class

    for SGD_iter in range(max_iters):
        """
        Do 'max_iters' iterations of sampling from prior,likelihood, and variational posterior,
        calculate ELBO gradient and update theta parameters
        """
        t +=1
        # Posterior must be updated when theta is updated (can do in var_posterior). Create set_theta function that you use in adam step

        # Draw z samples from variational posterior
        S0_samples, evals_samples, evecs_samples = var_posterior.rvs(size=K)
        D_samples = compute_D(evals_samples, evecs_samples)

        # Sample K samples of frozen prior, frozen likelihood and variational posterior
        log_prior_sample_list = np.zeros(K)
        log_likelihood_sample_list = np.zeros(K)
        log_var_post_sample_list = np.zeros(K)

        for i in range(K): # Number of samples drawn
            # Sample prior
            log_prior_sample_list[i] =fro_prior.logpdf(S0_samples[i], evals_samples[i][0],evals_samples[i][1], evals_samples[i][2], V=None)
            
            # Sample likelihood
            log_likelihood_sample_list[i] = fro_likelihood.logpdf(S0_samples[i], evecs_samples[i], evals_samples[i])

            # Sample variational posterior
            log_var_post_sample_list[i] = var_posterior.logpdf(S0_samples[i], D_samples[i])
        
        # Compute log p(D,z) for each sample
        log_p_of_D_and_z = log_likelihood_sample_list + log_prior_sample_list
        
        # Compute f(z) for each sample
        f_z_samples = log_p_of_D_and_z - log_var_post_sample_list

        # Estimate the gradient of ELBO using the REINFORCE leave-one-out estimator
        elbo_grad = 0
        for k in range(K):
            current = f_z_samples[k]
            others = (sum(f_z_samples) - current) / (K-1)
            elbo_grad += (current - others) * var_posterior.score(S0_samples[k], D_samples[k])
        elbo_grad /= K

        # Adam
        m = b1 * m + (1-b1) * elbo_grad
        v = b2 * v + (1-b2) * elbo_grad**2
        m_hat = m / (1-b1**t)
        v_hat = v / (1-b2**t)
        theta += learning_rate * (m_hat / (np.sqrt(v_hat) + e))

        # Update the posterior
        var_posterior.set_theta(theta)

    Sigma = D_from_theta(theta[2:8])
    theta_1 = theta[0]
    theta_2 = theta[1]
    theta_9 = theta[8]

    return variational_posterior(Sigma, theta_1, theta_2, theta_9)


@disk_memoize()
def laplace_approximation(y, gtab, prior=None, init=None, variance=29, bounds=None):
    raise NotImplementedError


"""
=============================================================================
Visualization & Experiment Runner
=============================================================================
Plotting function and the main() script to run experiments.
"""

def main():
    # Initialize with preprocessed data and DTI point estimate
    # (these values can be used as starting points for inference methods)
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=True)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Find principal eigenvector from DTI estimate (for plotting)
    evec_principal = evecs_init[:, 0]

    # Set random seed and number of posterior samples
    np.random.seed(0)
    n_samples = 10000

    """
    # Run Metropolis–Hastings and plot results
    S0_mh, evals_mh, evecs_mh = metropolis_hastings(force_recompute=False)
    burn_in = 2000
    plot_results(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:, :, :], evec_principal, method="mh")
    """
    '''
    # Run Importance Sampling and plot results
    w_is, S0_is, evals_is, evecs_is = importance_sampling(force_recompute=False)
    plot_results(S0_is, evals_is, evecs_is, evec_principal, weights=w_is, method="is")
    ''' 
    
    # Run Variational Inference and plot results
    posterior_vi = variational_inference(max_iters=1000, K=256, learning_rate=1e-4, force_recompute=False)
    S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=n_samples)
    plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi")
    
    '''
    # Run Laplace Approximation and plot results
    #S0, evals, evecs, y, point_estimate, gtab = get_preprocessed_data()
    init = {"S0": S0_init, "evals": evals_init, "evecs": evecs_init}

    posterior_laplace = laplace_approximation( y=y, gtab=gtab, init=init, force_recompute=False)
    S0_laplace, evals_laplace, evecs_laplace = posterior_laplace.rvs(size=n_samples)
    plot_results(S0_laplace, evals_laplace, evecs_laplace, evec_principal, method="laplace")
    '''
    print("Done.")


def plot_results(S0, evals, evecs, evec_ref, weights=None, method=""):
    """
    Plot posterior results as histograms and save to file.

    Creates histograms of baseline signal (S0), mean diffusivity (MD),
    fractional anisotropy (FA), and the angle between estimated and
    reference eigenvectors.

    Parameters
    ----------
    S0 : ndarray
        Sampled baseline signals.
    evals : ndarray
        Sampled eigenvalues of the diffusion tensor.
    evecs : ndarray
        Sampled eigenvectors of the diffusion tensor.
    evec_ref : ndarray
        Reference principal eigenvector (from point estimate).
    weights : ndarray, optional
        Importance weights for samples. Uniform if None.
    method : str
        Name of inference method (used in output filename).
    """
    
    # Use uniform weights if none provided
    if weights is None:
        weights = np.ones_like(S0)
        weights /= np.sum(weights)

    # Choose number of bins based on sample size
    n_bins = np.floor(np.sqrt(len(weights))).astype(int)

    # Squeeze arrays for plotting
    weights = weights.squeeze()
    S0 = S0.squeeze()
    md = dti.mean_diffusivity(evals).squeeze()
    fa = dti.fractional_anisotropy(evals).squeeze()

    # Compute acute angle between estimated and reference eigenvectors
    angle = 360/(2*np.pi) * np.arccos(np.abs(np.dot(evecs[:, :, 2], evec_ref)))
    
    # Create 2x2 grid of histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

    axes[0, 0].hist(S0, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel("S0")
    axes[0, 0].set_ylabel("Density")

    axes[0, 1].hist(md, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel("Mean diffusivity")
    axes[0, 1].set_ylabel("Density")

    axes[1, 0].hist(fa, bins=n_bins, density=True, weights=weights,
                     alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel("Fractional anisotropy")
    axes[1, 0].set_ylabel("Density")

    axes[1, 1].hist(angle, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_xlabel("Acute angle")
    axes[1, 1].set_ylabel("Density")

    # Adjust layout and save figure with method name
    plt.tight_layout()
    plt.savefig("results_{}.png".format(method), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()