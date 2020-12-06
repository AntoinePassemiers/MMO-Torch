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
Optimization over the manifold is enabled by passing it to
a Riemannian optimizer:
```python
optimizer = RiemannianSGD([W], lr=1e-2)
```

Perform inference, the usual way:
```python
X = torch.rand(100, 40)
Y = torch.rand(100, 40)

for _ in range(50):
    Y_hat = X.mm(W)
    loss = torch.mean((Y_hat - Y) ** 2.)
    loss.backward()
    optimizer.step()
    print(loss.item())
```


Installation
------------

### Dependencies


To get ArchMM to work on your computer, you will need:

- Python
- Numpy (>= 1.6.1)
- Torch

### User installation

Install the package :
```
python setup.py install
```
