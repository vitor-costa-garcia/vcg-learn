from .core import BaseOptimizer

class GradientDescent(BaseOptimizer):
	def step(self, W, gradient):
		return W - self._lr * gradient