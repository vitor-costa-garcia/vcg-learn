from ..core import BaseTransformer
from . import SVD

class PCA(BaseTransformer):
	def __init__(self, n_components):
		self._computed = True
		self._n_components = n_components

	def fit(self, X, y = None):
		#Block transform
		self._computed = False

		#Calculate the covariance matrix
		cov_mat = self._compute_covmat(X)

		#Calculate the SVD (Since covmat is symmetric, u = v.T)
		svd = SVD()
		U, S, V_t = svd.fit_transform(cov_mat)
		self._proj = V_t

		#Allow transform
		self._computed = True

	def transform(self, X, y = None):
		if self._computed:
			return self._proj[:self._n_components, :] @ X

	def _compute_covmat(self, X):
		return X @ X.T