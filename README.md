# frankwolfe-py

This package is a python wrapper for the package [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl)

## Usage 

Each objective function must be wrapped to avoid compatibility issues with Julia.

In each file, import wrapper and wrap the objective function f using f = wrapper.wrap_objective_function(f).

A simple example : 

```python
from frankwolfepy import frankwolfe
from frankwolfepy import wrapper
import numpy as np

def f(x): #objective function
    return np.linalg.norm(x)**2

f = wrapper.wrap_objective_function(f) #wrap the objective function

def grad(storage,x): #gradient computation
    for i in range(len(x)):
        storage[i] = x[i]

# Create the Linear Minimization Oracle
lmo_prob = frankwolfe.ProbabilitySimplexOracle(1)

# Compute first point
x0 = frankwolfe.compute_extreme_point(lmo_prob,np.zeros(5))

#Solve the optimisation problem using vanilla Frank-Wolfe algorithm
frankwolfe.frank_wolfe(
    f,
    grad,
    lmo_prob,
    x0,
    max_iteration=1000,
    line_search=frankwolfe.Agnostic(),
    verbose=True,
)
```

For documentation on FrankWolfe.jl : https://zib-iol.github.io/FrankWolfe.jl/dev/

