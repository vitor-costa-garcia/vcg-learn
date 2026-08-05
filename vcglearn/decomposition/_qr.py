from ..core import BaseTransformer
from abc import ABC, abstractmethod
import numpy as np

class QRMethod(ABC):
	@abstractmethod
	def decompose(self, X):
		pass

class GramSchmidt(QRMethod):
	"""
	Gram Schmidt process is a method of building an orthonormal basis from a set of vectors
	"""
	def decompose(self, X):
		"""
		Returns a orthonormal basis based off the columns of X
		"""
		X_c = X.copy()
		m, n = X.shape

		#For each columns	
		for i in range(1, n):
			v_n = X[:, i].copy()
			u_i = X[:, i].copy()

			#Subtract projection of all i-1 vectors
			for k in range(i):
				u_k = X_c[:, k]

				u_i -= (np.dot(v_n, u_k) / np.dot(u_k, u_k)) * u_k

			#Replace column with orthogonal vector
			X_c[:, i] = u_i

		#Normalize columns
		X_norms = np.linalg.norm(X_c, axis=0)
		Q = X_c / X_norms

		return Q, Q.T @ X

class Householder(QRMethod):
	"""
	Householder transformations take a vector and reflects it about some hyperplane or plane.
	"""
	def decompose(self, X):
		m, n = X.shape

		#Right upper triangle matrix R
		R = X.astype(float).copy()

		# Orthogonal matrix Q that satisfies A=QR
		Q = np.identity(m)

		#We use min(m, n) to make it valid for rectangular matrices.
		for i in range(min(m, n)):
			# Calculate H_i
			x = R[i:, i]
			h_small = self._calculate_h(x)

			#Embbed small H_i into identity matrix to makethe product of matrices valid
			h_i = np.identity(m)
			h_i[i:, i:] = h_small

			#Multiply R and Q
			R = h_i @ R
			Q = Q @ h_i

		return Q, R

	def _calculate_h(self, x) -> np.array:
		m = x.shape[0]

		e_1 = np.zeros(m)
		e_1[0] = 1.

		alpha = np.linalg.norm(x)

		#u vector
		sign = 1.0 if x[0] >= 0 else -1.0
		u = x + sign * np.linalg.norm(x) * e_1
		u_norm = np.linalg.norm(u)

		#v vector
		v = u / u_norm
		vtv = np.outer(v, v)

		# Householder transformation
		return np.identity(m) - 2 * vtv

class QRDecomposition(BaseTransformer):
	"""
	QR Decomposition is a popular decomposition in linear algebra that represents an NxM matrix
	into a product of an orthogonal matrix Q that satisfies A = QR and a upper right triangle matrix R
	(R = Q^TA). For computing Q, one can use GramSchmidt or Householder.
	"""
	def __init__(self, method: QRMethod = Householder()):
		self._computed = False
		self._method = method

	def fit(self, X, y = None):
		#Block get
		self._computed = False

		# Using QRMethod to calculate Q
		self._Q, self._R = self._calculate_qr(X)

		#Allow get
		self._computed = True

	def transform(self, X, y = None):
		if self._computed:
			return self._Q, self._R

	#Computing orthogonal matrix that satisfies A = QR
	def _calculate_qr(self, X):
		return self._method.decompose(X)