import numpy as np

from ..core import BaseTransformer

class SVD(BaseTransformer):
	"""
	SVD is a popular linear algebra method for decomposition, which consists in computing
	a representation of a matrix A in the form A = USV^T, where U and V are orthogonal matrices
	and S is a diagonal matrix of singular values.

	Methods:
	fit(X, y): Computes the SVD of a matrix X. (y has no utility here)
	predict(X, y) -> U, S, V^T: Returns the result of SVD 
	"""
	def __init__(self):
		self.computed = False

		#SVD
		self._U_r = None
		self._sigma_r = None
		self._V_r = None

		self._singval = None

	def fit(self, X, y = None):
		#Block predict method return until succesful SVD
		self.computed = False
		tol = 10e-5

		#Computing matrix A^TA
		ata = self._compute_ata(X)

		#Computing eigenvalues and orthonormal eigenvectors of A^TA (V)
		eigval, eigvec = self._compute_eigv(ata)
		eigval = self._flip_eigval(eigval)
		V = self._flip_eigvec(eigvec)

		#Removing negative eigenvalues caused by floating point error
		eigval = self._clip_eigval(eigval)

		#Computing singular values and diagonal sigular values matrix
		self._singval = self._compute_singv(eigval)

		mask = self._singval < tol

		#Cutting off invalid singular values equal to zero
		self._singval = self._tol_singv(self._singval, mask) 

		#Computing diagonal singular values matrix
		self._sigma_r = self._compute_sigma(self._singval)
		sigma_inv = self._compute_sigma(1 / self._singval)

		#Using only valid columns of V (columns which singular value is not 0)
		self._V_r = self._clip_v(V, mask)

		#Computing orthonormal eigenvectors of AA^T
		self._U_r = self._compute_u(X, self._V_r, sigma_inv)

		#Allow predict method return
		self.computed = True

	def transform(self, X = None, y = None):
		if self.computed:
			return self._U_r, self._sigma_r, self._V_r.T

	# --------------------------------------------------------------------------
	# Computing A^TA
	def _compute_ata(self, A):
		return A.T @ A

	# Flip eigenvalue vector to ascending order
	def _flip_eigval(self, A):
		return np.flip(A)

	# Flip orthogonal eigenvector matrix to match eigenvalue order
	def _flip_eigvec(self, A):
		return np.flip(A, axis=1)

	def _clip_eigval(self, A):
		return np.clip(A, 0, None)

	# Computing eigenvalues of matrix S
	def _compute_eigv(self, S):
		return np.linalg.eigh(S)

	# Computing singular vectors from eigenvalues A
	def _compute_singv(self, A):
		return np.sqrt(A)

	# Apply a tolerance of 10-e5 to avoid precision error
	def _tol_singv(self, A, tol_mask):
		return A[~tol_mask]

	# Clipping V to get valid singular value orthonormal vectors only
	def _clip_v(self, A, tol_mask):
		return A[:, ~tol_mask]

	# Compute matrix U of orthonormal vectors
	def _compute_u(self, A, V, S):
		return A @ V @ S

	# Computing diagonal singular vector matrix from singular values A
	def _compute_sigma(self, A):
		return np.diag(A)

