import numpy as np
from scipy import linalg
from sklearn.datasets import make_spd_matrix
import scipy
import joblib
from sklearn.mixture import GaussianMixture as GMM
import torch
import os


def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))


def load_or_create_data_gmm(n_components=None, n_dim=None, zeromean=None,
                                num_train_samples=None, num_test_samples=None, num_val_samples=None, seed=352634,
            return_gmm=False, return_torch=False, normalize=False, dataset='rand_gmm', mode='1D', make_zero_mean=True, norm_per_dim=True, dtype='real'):
    os.makedirs(f'./bin/{dataset}/', exist_ok=True)
    if dataset == 'rand_gmm' or dataset == 'audio_gmm':
        filename_gmm = f'./bin/{dataset}/gmm_dim={n_dim}_zeromean={zeromean}_comp={n_components}_{dtype}_seed={seed}.sav'
        filename_train = f'./bin/{dataset}/data={num_train_samples}_dim={n_dim}_zeromean={zeromean}_comp={n_components}_{dtype}_data_train_seed={seed}.npy'
        filename_test = f'./bin/{dataset}/data={num_test_samples}_dim={n_dim}_zeromean={zeromean}_comp={n_components}_{dtype}_data_test_seed={seed}.npy'
        filename_val = f'./bin/{dataset}/data={num_val_samples}_dim={n_dim}_zeromean={zeromean}_comp={n_components}_{dtype}_data_val_seed={seed}.npy'
        try:
            data_train = np.load(filename_train)
            data_test = np.load(filename_test)
            data_val = np.load(filename_val)
            gmm_rand = joblib.load(filename_gmm)
        except FileNotFoundError:
            n_data = num_train_samples + num_val_samples + num_test_samples
            rng = np.random.default_rng(seed)
            gmm_rand = GMM(n_components=n_components)
            gmm_rand_init(gmm=gmm_rand, dim=n_dim, zeromean=zeromean, rand_state=seed)
            joblib.dump(gmm_rand, filename_gmm)
            data, comps = gmm_rand.sample(n_data)
            rng.shuffle(data, axis=0)
            rng.shuffle(comps, axis=0)
            data_train = data[:num_train_samples]
            data_test = data[num_train_samples:num_train_samples + num_test_samples]
            data_val = data[-num_val_samples:]
            del data
            np.save(filename_train, data_train)
            np.save(filename_test, data_test)
            np.save(filename_val, data_val)
    else:
        if dataset == 'FASHION_MNIST_gmm' or dataset == 'MNIST_gmm':
            n_dim = 28**2
        filename_gmm = f'./bin/{dataset}/gmm_zeromean={zeromean}_comp={n_components}_seed={seed}.sav'
        filename_train = f'./bin/{dataset}/data={num_train_samples}_zeromean={zeromean}_comp={n_components}_train_seed={seed}.npy'
        filename_test = f'./bin/{dataset}/data={num_test_samples}_zeromean={zeromean}_comp={n_components}_test_seed={seed}.npy'
        filename_val = f'./bin/{dataset}/data={num_val_samples}_zeromean={zeromean}_comp={n_components}_val_seed={seed}.npy'
        data_train = np.load(filename_train)
        data_test = np.load(filename_test)
        data_val = np.load(filename_val)
        gmm_rand = joblib.load(filename_gmm)
    if normalize:
        if make_zero_mean:
            mu_train = np.mean(data_train, axis=0)
            data_train -= mu_train[None, :]
            data_test -= mu_train[None, :]
            data_val -= mu_train[None, :]
        else:
            mu_train = np.zeros_like(data_train[0], dtype=data_train.dtype)
        if norm_per_dim:
            squared_norm = np.var(data_train, axis=0)
            norm_fac = np.sqrt(squared_norm)
        else:
            squared_norm = np.mean(np.linalg.norm(data_train, axis=1)**2)
            norm_fac = np.sqrt(squared_norm / n_dim)
        data_train = data_train / norm_fac
        data_test = data_test / norm_fac
        data_val = data_val / norm_fac
    if mode == '2D':
        dim_1d = int(np.sqrt(data_train.shape[-1]))
        data_train = np.reshape(data_train, (-1, dim_1d, dim_1d), 'F')
        data_test = np.reshape(data_test, (-1, dim_1d, dim_1d), 'F')
        data_val = np.reshape(data_val, (-1, dim_1d, dim_1d), 'F')
    if return_torch:
        if dtype == 'real':
            data_train = torch.from_numpy(data_train[:, None, :]).float()
            data_test = torch.from_numpy(data_test[:, None, :]).float()
            data_val = torch.from_numpy(data_val[:, None, :]).float()
        else:
            data_train = np.concatenate((np.real(data_train[:, None, :]),np.imag(data_train[:, None, :])), axis=1)
            data_test = np.concatenate((np.real(data_test[:, None, :]),np.imag(data_test[:, None, :])), axis=1)
            data_val = np.concatenate((np.real(data_val[:, None, :]),np.imag(data_val[:, None, :])), axis=1)
            data_train = torch.from_numpy(data_train).float()
            data_test = torch.from_numpy(data_test).float()
            data_val = torch.from_numpy(data_val).float()
    if return_gmm:
        if normalize:
            normalize_gmm(gmm_rand, norm_fac=norm_fac, shift=mu_train, norm_per_dim=norm_per_dim, dtype=dtype)
        return data_train, data_val, data_test, gmm_rand
    else:
        return data_train, data_val, data_test


def gmm_rand_init(gmm, dim, zeromean=False, rand_state=1244563):
    rng = np.random.default_rng(rand_state)
    gmm.covariances_ = np.zeros([gmm.n_components, dim, dim])
    for i, comp in enumerate(range(gmm.n_components)):
        rand_state_covs = rand_state + i
        gmm.covariances_[comp] = make_spd_matrix(dim, random_state=rand_state_covs)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, 'full')
    gmm.weights_ = rng.uniform(0, 1, gmm.n_components)
    gmm.weights_ /= np.sum(gmm.weights_)
    if zeromean:
        gmm.means_ = np.zeros_like(gmm.covariances_[0])
    else:
        gmm.means_ = rng.standard_normal((gmm.n_components, dim)) / np.sqrt(dim)


def normalize_gmm(gmm, norm_fac: float=1.0, shift: np.ndarray=0.0, norm_per_dim=False, dtype='True'):
    if dtype == 'real':
        gmm.means_ -= shift
        gmm.means_ /= norm_fac
        if norm_per_dim:
            D = np.diag(1 / norm_fac)
            gmm.covariances_ = D[None, :, :] @ gmm.covariances_ @ D[None, :, :]
        else:
            gmm.covariances_ /= norm_fac**2
    else:
        NotImplementedError('Normalization for complex GMM not implemented.')


def compute_expected_squared_norm(gmm):
    """
    Computes E[||x||^2] = trace(C_global) + trace(mu_global @ mu_global^H)
    """
    glob_mean = np.zeros_like(gmm.means_[0])
    glob_cov = np.zeros_like(gmm.covariances_[0])
    for k in range(gmm.n_components):
        glob_mean += gmm.weights_[k] * gmm.means_[k]
    for k in range(gmm.n_components):
        glob_cov += gmm.weights_[k] * (gmm.covariances_[k] + gmm.means_[k, :, None] @ gmm.means_[k, None].conj())
    glob_cov -= glob_mean[:, None] @ glob_mean[:, None].T.conj()
    squared_norm = np.trace(glob_cov) + np.trace(glob_mean[:, None] @ glob_mean[:, None].T.conj())
    return squared_norm


def prepare_for_prediction(gmm, sigma2):
    for k in range(gmm.n_components):
        gmm.covariances_[k] = gmm.covariances_[k] + sigma2 * np.eye(gmm.covariances_.shape[-1])
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, 'full')
    Cy_inv = np.linalg.pinv(gmm.covariances_)
    return Cy_inv


def lmmse_formula(gmm, Cy_inv, k, y):
    return gmm.means_[k] + gmm.covariances_[k] @ Cy_inv[k] @ (y - gmm.means_[k])



class Rand_GMM:
    def __init__(self, comp, dim, zeromean=False, real=True, rand_state=151651):
        self.comp = comp
        self.dim = dim
        self.zeromean = zeromean
        if real:
            self.dtype = float
        else:
            raise NotImplementedError
        self.cov = np.zeros([comp, dim, dim], dtype=self.dtype)
        self.cov_sqrt = np.zeros_like(self.cov)
        self.cov_sqrt = np.zeros([comp, dim, dim], dtype=self.dtype)
        self.mu = np.zeros([comp, dim], dtype=self.dtype)
        self.weights = np.zeros([comp])
        self.rng = np.random.default_rng(rand_state)
        self.rand_state = rand_state
        self.glob_mean = np.zeros_like(self.mu[0])
        self.glob_cov = np.zeros_like(self.cov[0])
        self.squared_norm = None


    def random_init(self):
        for comp in range(self.comp):
            self.cov[comp] = make_spd_matrix(self.dim, random_state=self.rand_state)
            self.cov_sqrt[comp] = scipy.linalg.sqrtm(self.cov[comp])
        self.weights = self.rng.uniform(0, 1, self.comp)
        self.weights /= np.sum(self.weights)
        if not self.zeromean:
            self.mu = self.rng.standard_normal((self.comp, self.dim))


    def sample(self, n_samp):
        n_samples_com = self.rng.multinomial(n_samp, self.weights)
        x = np.zeros([n_samp, self.dim], dtype=self.dtype)
        index = 0
        for comp in range(self.comp):
            for data in range(n_samples_com[comp]):
                x[index] = np.squeeze(self.cov_sqrt[comp] @  self.rng.standard_normal((self.dim, 1))) + self.mu[comp]
                index += 1
        return x


    def compute_global_cov_mean(self):
        for k in range(self.comp):
            self.glob_mean += self.weights[k] * self.mu[k]
        for k in range(self.comp):
            self.glob_cov += self.weights[k] * (self.cov[k] + self.mu[k, :, None] @ self.mu[k, None].conj())
        self.glob_cov -= self.glob_mean[:, None] @ self.glob_mean[:, None].T.conj()


    def compute_expected_squared_norm(self):
        """
        Computes E[||x||^2] = trace(C_global) + trace(mu_global @ mu_global^H)
        """
        self.squared_norm = np.trace(self.glob_cov) + np.trace(self.glob_mean[:, None] @ self.glob_mean[:, None].T.conj())



def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def rand_exp(left: float, right: float, shape: tuple[int, ...]=(1,), seed=None):
    r"""For 0 < left < right draw uniformly between log(left) and log(right)
    and exponentiate the result.

    Note:
        This procedure is explained in
            "Random Search for Hyper-Parameter Optimization"
            by Bergstra, Bengio
    """
    if left <= 0:
        raise ValueError('left needs to be positive but is {}'.format(left))
    if right <= left:
        raise ValueError(f'right needs to be larger than left but we have left: {left} and right: {right}')
    rng = np.random.default_rng(seed)
    return np.exp(np.log(left) + rng.random(*shape) * (np.log(right) - np.log(left)))