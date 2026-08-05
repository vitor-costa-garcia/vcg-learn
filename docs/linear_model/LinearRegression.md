# Linear Regression
## vcglearn.linear_model.LinearRegression

---

### Description

Linear regression consists on fitting a line that minimizes the MSE. Can be fitted using LeastSquares.

---

### Methods
 - fit(X, y = None): Fit method adjusts a line to X, minimizing MSE.
 	- X: Data matrix
 	- y: Label vector

 - predict(X, y = None) -> y_hat: Returns predictions for each datapoint of X. CANNOT BE CALLED BEFORE fit()
 	- X: Data matrix
 	- y: Not used

 - fit_predict(X, y) -> Fits a line to X, then returns predictions for each datapoint of X. 
 	- X: Data matrix
 	- y: Label vector

---

### Attributes

 - self._w: Adjusted coefficients vector w/ intercept
---

### Examples

```python
from vcglearn.decomposition import QRDecomposition
import numpy as np

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

print(weights, predictions)

# Results --------

# weights
# [1. 2.]

# predictions
# [13. 15. 17.]
```