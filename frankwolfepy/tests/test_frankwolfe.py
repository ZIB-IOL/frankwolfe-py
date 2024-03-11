from frankwolfepy import frankwolfe
from frankwolfepy import wrapper
import numpy as np

def test_simple_frankwolfe():
    
    def f(x):
        return frankwolfe.norm(x)**2

    def grad(storage,x):
        for i in range(len(x)):
            storage[i] = x[i]

    lmo_prob = frankwolfe.ProbabilitySimplexOracle(1)
    x0 = frankwolfe.compute_extreme_point(lmo_prob,np.zeros(5))
    
    assert(frankwolfe.frank_wolfe(wrapper.wrap_obj_func(f),grad,lmo_prob,x0,max_iteration=1000,line_search=frankwolfe.Agnostic(),verbose=True,)[3] - 0.2 < 1.0e-5 )

def test_lazified_cond_grad():

    def f(x):
        return frankwolfe.norm(x)*frankwolfe.norm(x)

    def grad(storage,x):
        for i in range(len(x)):
            storage[i] = x[i]

    lmo_prob = frankwolfe.ProbabilitySimplexOracle(1)
    x0 = frankwolfe.compute_extreme_point(lmo_prob,np.zeros(5))
    
    
    assert(frankwolfe.lazified_conditional_gradient(wrapper.wrap_obj_func(f),grad,lmo_prob,x0,max_iteration=1000,verbose=True,)[3] - 0.2 < 1.0e-5 )



def test_blended_cg():
    

    n = int(1e3)
    k = int(1e4)

    matrix = np.random.randn(n, n)
    hessian = np.dot(np.transpose(matrix),matrix)
    linear = np.random.rand(n)
    
    def f(x):
        return np.dot(linear, x) + 0.5 * np.dot(np.matmul(np.transpose(x),hessian),x)
    
    
    def grad(storage, x):
        storage[:] = np.add(linear,np.matmul(hessian,np.transpose(x)))
    
    f = wrapper.wrap_obj_func(f)

    L = max(np.linalg.eigvals(hessian))

    lmo = frankwolfe.ProbabilitySimplexOracle(1.0)
    x0 = frankwolfe.compute_extreme_point(lmo, np.zeros(n))
    

    target_tolerance = 1e-5

    x, v, primal, dual_gap, trajectoryBCG_accel_simplex, _ = frankwolfe.blended_conditional_gradient(
        f,
        grad,
        lmo,
        x0,
        epsilon=target_tolerance,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=L),
        print_iter=k / 10,
        hessian=hessian,
        memory_mode=frankwolfe.InplaceEmphasis(),
        accelerated=True,
        verbose=True,
        trajectory=True,
        lazy_tolerance=1.0,
        weight_purge_threshold=1e-10,
    )

    
    x, v, primal, dual_gap, trajectoryBCG_simplex, _ = frankwolfe.blended_conditional_gradient(
        f,
        grad,
        lmo,
        x0,
        epsilon=target_tolerance,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=L),
        print_iter=k / 10,
        hessian=hessian,
        memory_mode=frankwolfe.InplaceEmphasis(),
        accelerated=False,
        verbose=True,
        trajectory=True,
        lazy_tolerance=1.0,
        weight_purge_threshold=1e-10,
    )

    
    x, v, primal, dual_gap, trajectoryBCG_convex, _ = frankwolfe.blended_conditional_gradient(
        f,
        grad,
        lmo,
        x0,
        epsilon=target_tolerance,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=L),
        print_iter=k / 10,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        trajectory=True,
        lazy_tolerance=1.0,
        weight_purge_threshold=1e-10,
    )



def test_away_step_frankwolfe():
    
    n = int(1e4)
    k = int(2e3)
    number_nonzero = 40

    xpi = np.random.rand(n)
    total = sum(xpi)
    xp = xpi

    def f(x):
        return np.linalg.norm(x - xp)**2
    
    def grad(storage, x):
        storage[:] = 2 * (x - xp)

    f = wrapper.wrap_obj_func(f)

    lmo = frankwolfe.KSparseLMO(number_nonzero, 1.0)

    x0 = frankwolfe.compute_extreme_point(lmo, np.ones(n))
    
    x, v, primal, dual_gap, trajectory, active_set = frankwolfe.away_frank_wolfe(
        f,
        grad,
        lmo,
        x0,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(),
        print_iter=k / 10,
        epsilon=1e-5,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        trajectory=True,
        lazy=True,
    )

    x, v, primal, dual_gap, trajectoryAFW, active_set = frankwolfe.away_frank_wolfe(
        f,
        grad,
        lmo,
        x0,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(),
        print_iter=k / 10,
        epsilon=1e-5,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        trajectory=True,
        away_steps = True,
    )

    x, v, primal, dual_gap, trajectoryFW , active_set = frankwolfe.away_frank_wolfe(
        f,
        grad,
        lmo,
        x0,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(),
        print_iter=k / 10,
        epsilon=1e-5,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        trajectory=True,
        away_steps=False,
    )

