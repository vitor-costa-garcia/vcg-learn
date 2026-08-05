# Matrix inversion
# vcglearn.utils.linalg.invert

---
### Description

Inverts a square matrix. 

---

### Parameters

 - X: Invertible square matrix

---

### Examples

```python
from vcglearn.utils.linalg import invert
import numpy as np

A = np.array([
	[3, 0],
	[0, 2],
], dtype=float)

A_inv = invert(A)

print(A, A_inv, A@A_inv)

# Results --------

# A
# [3. 0.]
# [0. 2.]
  
# A_inv
# [ 0.33333333 -0.        ]
# [-0.          0.5       ]

# A@A_inv
# [1. 0.]
# [0. 1.]
```