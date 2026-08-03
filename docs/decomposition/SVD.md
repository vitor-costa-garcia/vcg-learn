# Singular Vector Decomposition
## vcglearn.decomposition.SVD 

### Description
SVD (Singular Vector Decomposition) is a matrix decomposition method used to transform a NxM matrix into a product of 2 orthogonal matrices and one diagonal singular values matrix.

### Methods

 - fit(X, y = None): Fit method performs the SVD on the matrix X and saves the result.
 	- X: Target matrix that will be used in SVD
 	- y: Not used

 - predict(X = None, y = None) -> U, D, V^T: Returns the result of the SVD. CANNOT BE CALLED BEFORE fit()
 	- X: Not used
 	- y: Not used

 - fit_predict(X, y = None) -> U, D, V^T: Performs and returns the resul of the SVD.
 	- X: Target matrix that will be used in SVD
 	- y: Not used

### Attributes

 - self._singval: Singular values in ascending order.
 - self._U: Orthonormal eigenvectors of A^TA.
 - self._sigma: Diagonal singular value matrix.
 - self._V: Orthonormal eigenvectors of AA^T