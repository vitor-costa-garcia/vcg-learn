from ..core import BaseEstimator
from ..utils.linalg import invert
from abc import ABC, abstractmethod
import numpy as np

class Solver(ABC):
	@abstractmethod
	def adjust(self, X, y):
		raise NotImplementedError



class LeastSquares(Solver):
	def adjust(self, X, y):
		XTy = self._calculate_xty(X, y)

		XTX = self._calculate_xtx(X)
		XTX_inv = invert(XTX)

		w = self._calculate_w(XTX_inv, XTy)

		return w

	#Least-Square functions
	def _calculate_xtx(self, X):
		return X.T @ X

	def _calculate_xty(self, X, y):
		return X.T @ y

	def _calculate_w(self, XTX_inv, XTy):
		return XTX_inv @ XTy



class LinearRegression(BaseEstimator):
	"""
	Linear regression is a popular method to fit a linear equation to a collection of points.
	
	Parameters:
	method: Fitting method [LeastSquares(), GradientDescent()]
	"""

	def __init__(self, method: Solver = LeastSquares()):
		self._method = method
		self._computed = False

	def fit(self, X, y):
		#Block predict
		self._computed = False

		#Add column for intercept fitting
		X_one = self._add_one_col(X)

		#Fit line
		self._w = self._method.adjust(X_one, y)

		#Allow predict
		self._computed = True

	def predict(self, X, y = None):
		if self._computed:
			#Dot product between X and w returns y_hat
			X_one = self._add_one_col(X)
			return X_one @ self._w

	def _add_one_col(self, X):
		m, n = X.shape
		X_one = np.column_stack((np.ones(m), X))

		return X_one