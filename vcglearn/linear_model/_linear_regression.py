from ..core import BaseEstimator

class LinearRegression(BaseEstimator):
	"""
	Linear regression is a popular method to fit a linear equation to a collection of points.
	
	Parameters:
	method: Fitting method ['lstsq', 'gd']
	"""
	def __init__(self, method="lstsq"):
		self._method = method

	def fit(self, X, y):
		#Calculate X^Ty
		xtx = self._calculate_xtx(X)

		#Calculate X^TX
		xty = self._calculate_xty(X, y)
		#Invert X^TX
		#Calculate w = (X^TX)^-1 X^y


	def predict(self, X, y):
		#Dot product between X and w returns y_hat

	#Least-Square functions
	def _calculate_xty(X):
		return X.T @ X

	def _calculate_xtx(X, y):
		return X.T @ y

	def _invert_xtx():
		pass

	def _calculate_w():
		pass