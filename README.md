# vcg-learn

---

Python Machine Learning Framework

## Getting started

1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run tests (Optional)

```bash
python -m pytest
```

## Examples

```python
from vcglearn.decomposition import SVD
import numpy as numpy

A = np.array([
	[3, 1],
	[1, 2],
	[2, 1],
], dtype=float)

svd = SVD()
u, d, v = svd.fit_predict(A)
```