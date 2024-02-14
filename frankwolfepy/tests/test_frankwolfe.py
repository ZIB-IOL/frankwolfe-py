from frankwolfepy import frankwolfe
import numpy as np

def test_simple_frankwolfe():
    
    def f(x):
        return frankwolfe.norm(x)*frankwolfe.norm(x)

    def grad(storage,x):
        for i in range(len(x)):
            storage[i] = x[i]

    lmo_prob = frankwolfe.ProbabilitySimplexOracle(1)
    x0 = frankwolfe.compute_extreme_point(lmo_prob,np.zeros(5))
    
    assert(frankwolfe.frank_wolfe(f,grad,lmo_prob,x0,max_iteration=1000,line_search=frankwolfe.Agnostic(),verbose=False,)[3] - 0.2 < 1.0e-5 )
    