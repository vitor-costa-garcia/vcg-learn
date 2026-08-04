from ..core import BaseTransformer
from abc import ABC, abstractmethod
import numpy as np

class QRMethod(ABC):
	@abstractmethod
	def decompose(self, X):
		pass

class GramSchmidt(QRMethod):
	def decompose(self, X):
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

		return Q

class Householder(QRMethod):
	def decompose(self, X):
		pass

class QRDecomposition(BaseTransformer):
	def __init__(self, method: QRMethod = GramSchmidt()):
		self._computed = False
		self._method = method

	def fit(self, X, y = None):
		#Block get
		self._computed = False

		# Using QRMethod to calculate Q
		self._Q = self._calculate_q(X)

		# Calculate R = Q^TA
		self._R = self._calculate_r(self._Q, X)

		#Allow get
		self._computed = True

	def transform(self, X, y = None):
		if self._computed:
			return self._Q, self._R

	#Computing orthogonal matrix that satisfies A = QR
	def _calculate_q(self, X):
		return self._method.decompose(X)

	#Computing upper triangular matrix R
	def _calculate_r(self, Q, A):
		return Q.T @ A