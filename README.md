# Matrix Manifold Optimization with PyTorch

Initialize a manifold:
```python
from mmotorch.manifolds import StiefelManifold

manifold = StiefelManifold(40, 40)
```

Create a random matrix on the manifold:
```python
W = manifold.init()
```

The Manifold.init() method returns a torch.nn.Parameter object.
Actual optimization over the manifold is enabled by passing it to
a Riemannian optimizer:
```python
from mmotorch.optim import RiemannianSGD

optimizer = RiemannianSGD([W], lr=1e-3)
```

Perform inference, the usual way. Following example looks for the orthogonal projection
that minimizes mean squared error.
```python
import torch

X = torch.rand(100, 40)
Y = torch.rand(100, 40)

for _ in range(50):
	optimizer.zero_grad()
    Y_hat = X.mm(W)
    loss = torch.mean((Y_hat - Y) ** 2.)
    loss.backward()
    optimizer.step()
    print(loss.item())
```


Installation
------------

### Dependencies


To get MMO-Torch to work on your computer, you will need:

- Python
- Numpy (>= 1.6.1)
- Torch

### User installation

Install the package :
```
python setup.py install
```

TODO
----

Only few manifolds are supported at the moment.
More will be coming in the future.
