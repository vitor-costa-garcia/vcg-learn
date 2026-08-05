from abc import ABC, abstractmethod

class BaseClass(ABC):
	"""
	BaseClass class defines a deafult interface that all base classes must follow in this framework
	"""
	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def fit(self, X, y):
		pass

class BaseEstimator(BaseClass):
	"""
	BaseEstimator class defines a default interface that estimators must follow in this framework
	"""

	@abstractmethod
	def predict(self, X, y):
		pass

	def fit_predict(self, X, y):
		self.fit(X, y)
		return self.predict(X, y)

class BaseTransformer(BaseClass):
	"""
	BaseTransformer class defines a default interface that transformers must follow in this framework
	"""
	@abstractmethod
	def transform(self, X, y=None):
		pass

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X, y)

class BaseOptimizer(ABC):
	"""
	BaseOptimizer class defines a default interface that optimizers must follow in this framework
	"""
	@abstractmethod
	def __init__(self, lr):
		self._lr = lr

	@abstractmethod
	def step(self, W, gradient):
		raise NotImplementedError