import numpy as np
from ...decomposition import QRDecomposition

def invert(X):
	"""
	Inverts a matrix using QR Decomposition and back-substitution
	"""
	qr = QRDecomposition()
	Q, R = qr.fit_transform(X)

	A_inv = _back_subst(R, Q.T[:, 0])

	n, _ = Q.shape

	for i in range(1, n):
		a_i = _back_subst(R, Q.T[:, i])
		A_inv = np.column_stack((A_inv, a_i))

	return A_inv


def _back_subst(R, b):
	"""Performs back-substitution to solve Rx=b, where R is a upper right triangle matrix"""
	x = np.zeros_like(b)
	n = len(b)

	for i in range(n-1, -1, -1):

		sum_prev = 0
		for j in range(i+1, n):
			sum_prev += R[i, j] * x[j]

		x[i] = (b[i] - sum_prev) / R[i, i]

	return x