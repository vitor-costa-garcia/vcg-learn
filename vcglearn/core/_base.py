from abc import ABC, abstractmethod

class BaseEstimator(ABC):
	"""
	BaseEstimator class defines a default interface that estimators must follow in this framework
	"""
	def __init__(self):
		pass

	@abstractmethod
	def fit(self, X, y):
		pass

	@abstractmethod
	def predict(self, X, y):
		pass

	def fit_predict(self, X, y):
		self.fit(X, y)
		return self.predict(X, y)
