# Principal Component Anlysis
## vcglearn.decomposition.PCA
---
### Description
PCA (Principal Component Analysis) is a dimension reduction that uses the principal components of the covariance matrix to find the directions in which data has the most variance, so one can project their data to a lower dimension space with minimal variance loss.
---
### Parameters
 - n_components: Number of k principal components that will be used to project X into k dimensions.

---
### Methods
 - fit(X, y = None): Fit method computes k principal components on the covariance matrix of X and saves the result.
 	- X: Target matrix that will be used in PCA
 	- y: Not used

 - transform(X = None, y = None) -> X_k: Returns the result of the data projected into a lower space of dimension k. CANNOT BE CALLED BEFORE fit()
 	- X: Not used
 	- y: Not used

 - fit_predict(X, y = None) -> X_k: Performs and returns the result of the PCA.
 	- X: Target matrix that will be used in PCAS
 	- y: Not used
---
### Attributes

 - self._proj: Projection matrix
 - self._explained_var: Singular values of the covariance matrix
---
### Examples

```python
from vcglearn.decomposition import PCA
import numpy as numpy

X = np.array([
    [1, 2, 3],
    [1, 2, 3]
], dtype=float)

pca = PCA(n_components=1)
pca.fit(X)

X_reduced = pca.transform(X)

print(X_reduced)
print(pca._explained_var)

# Results --------
# X_reduced
# [[1.41421356 2.82842712 4.24264069]]

#pca._explained_va
# [28.]
```