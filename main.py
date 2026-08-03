from vcglearn.decomposition import SVD
import numpy as np



if __name__ == "__main__":

	A = np.array([
	    [1, 2, 3],
	    [2, 4, 6],
	    [3, 6, 9],
	    [4, 8, 12]
	], dtype=float)

	print(A.T @ A)
	print("Initial matrix:\n", A, "\n")
	svd = SVD()
	y = svd.fit_predict(A, 0)
	print(y)