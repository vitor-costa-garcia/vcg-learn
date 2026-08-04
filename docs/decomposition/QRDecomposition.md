# QR Decomposition
## vcglearn.decomposition.QRDecomposition
---
### Description
QR Decomposition is a matrix decomposition method used to transform a NxM matrix into a product a orthogonal matrix Q and a upper triangle matrix R.
---
### Methods
 - fit(X, y = None): Fit method performs the QR decomposition on the matrix X and saves the result.
 	- X: Target matrix that will be used in QR decomposition
 	- y: Not used

 - transform(X, y = None) -> Q, R: Returns the result of the QR decomposition. CANNOT BE CALLED BEFORE fit()
 	- X: Not used
 	- y: Not used

 - fit_transform(X, y = None) -> Q, R: Performs and returns the resul of the QR decomposition.
 	- X: Target matrix that will be used in QR decomposition
 	- y: Not used
---
### Attributes

 - self._Q: Orthogonal matrix Q that satisfies A = QR.
 - self._R: Upper right triangle matrix R (R = Q^TA).
---
### Examples

```python
from vcglearn.decomposition import QRDecomposition
import numpy as np

A = np.array([
	[3, 1],
	[1, 2],
	[2, 1],
], dtype=float)

qr = QRDecomposition()
q, r = qr.fit_transform(A)

print(q, r, q@r)

# Results --------

# q
# [ 0.80178373 -0.31622777]
# [ 0.26726124  0.9486833 ]
# [ 0.53452248  0.        ]

# r
# [3.74165739 1.87082869]
# [0.         1.58113883]

# q@r
# [3. 1.]
# [1. 2.]
# [2. 1.]
```