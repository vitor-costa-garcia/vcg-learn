import numpy.linalg as np

from ..core import BaseEstimator

class SVD(BaseEstimator):
	def __init__(self):
		pass

	def fit(self, X, y):
		ata = X.T @ X
		return ata

	def predict(self, X, y):
		pass

	#Private methods
	def _compute_ata(X):
		return X.T @ X