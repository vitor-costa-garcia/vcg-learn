import numpy as np

from ..core import BaseEstimator, InputPlaceHolder

class SVD(BaseEstimator):
	"""
	SVD is a popular linear algebra method for decomposition, which consists in computing
	a representation of a matrix A in the form A = USV^T, where U and V are orthogonal matrices
	and S is a diagonal matrix of singular values.

	fit(X, y): Computes the SVD of a matrix X. (y has no utility here)
	predict(X, y) -> U, S, V^T: Returns the result of SVD 
	"""
	def __init__(self):
		self.computed = False

		#SVD
		self._U = None
		self._sigma = None
		self._V = None

		self._singval = None

	def fit(self, X, y = InputPlaceHolder()):
		#Block predict method return until succesful SVD
		self.computed = False

		#Computing matrix A^TA
		ata = self._compute_ata(X)

		#Computing eigenvalues and orthonormal eigenvectors of A^TA (V)
		eigval, eigvec = self._compute_eigv(ata)
		eigval = self._flip_eigval(eigval)
		self._V = self._flip_eigvec(eigvec)

		#Computing singular values and diagonal sigular values matrix
		self._singval = self._compute_singv(eigval)
		self._sigma = self._compute_sigma(self._singval)
		sigma_inv = self._compute_sigma(1 / self._singval)

		#Computing orthonormal eigenvectors of AA^T
		self._U = X @ self._V @ sigma_inv

		#Allow predict method return
		self.computed = True

	def predict(self, X = InputPlaceHolder(), y = InputPlaceHolder()):
		if self.computed:
			return self._U, self._sigma, self._V.T

	#Private methods ----------
	# Computing A^TA
	def _compute_ata(self, A):
		return A.T @ A

	# Flip eigenvalue vector to ascending order
	def _flip_eigval(self, A):
		return np.flip(A)

	# Flip orthogonal eigenvector matrix to match eigenvalue order
	def _flip_eigvec(self, A):
		return np.flip(A, axis=1)

	# Computing eigenvalues of matrix S
	def _compute_eigv(self, S):
		return np.linalg.eigh(S)

	# Computing singular vectors from eigenvalues A
	def _compute_singv(self, A):
		return np.sqrt(A)

	# Computing diagonal singular vector matrix from singular values A
	def _compute_sigma(self, A):
		return np.diag(A)

