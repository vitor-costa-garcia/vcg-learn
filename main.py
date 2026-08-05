from vcglearn.linear_model import LinearRegression, LeastSquares
import numpy as np



if __name__ == "__main__":
	X = np.array([
	    [1.0],
	    [2.0],
	    [3.0],
	    [4.0],
	    [5.0]
	])

	y = np.array([
	    3.0,
	    5.0,
	    7.0,
	    9.0,
	    11.0
	])

	model = LinearRegression(method=LeastSquares())

	model.fit(X, y)

	X_test = np.array([
	    [6.0],
	    [7.0],
	    [8.0]
	])

	predictions = model.predict(X_test)
	weights = model._w

	print(predictions, weights)