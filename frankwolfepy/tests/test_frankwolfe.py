from frankwolfepy import frankwolfe
from frankwolfepy import wrapper
import numpy as np
import copy

def test_simple_frankwolfe():

    n = int(1e2)
    k = int(1e4)
    
    def f(x):
        return np.linalg.norm(x)**2

    def grad(storage,x):
        for i in range(len(x)):
            storage[i] = x[i]

    lmo_prob = frankwolfe.ProbabilitySimplexOracle(1)
    x0 = frankwolfe.compute_extreme_point(lmo_prob,np.zeros(n))
    
    assert(frankwolfe.frank_wolfe(wrapper.wrap_objective_function(f),grad,lmo_prob,x0,max_iteration=k,line_search=frankwolfe.Agnostic(),verbose=True,)[3] - 0.2 < 1.0e-5 )

def test_lazified_cond_grad():

    n = int(1e2)
    k = int(1e4)

    def f(x):
        return np.linalg.norm(x)**2

    def grad(storage,x):
        storage[:] = x

    lmo_prob = frankwolfe.ProbabilitySimplexOracle(1)
    x0 = frankwolfe.compute_extreme_point(lmo_prob,np.zeros(n))
    
    
    assert(frankwolfe.lazified_conditional_gradient(wrapper.wrap_objective_function(f),grad,lmo_prob,x0,max_iteration=k,verbose=True,)[3] - 0.2 < 1.0e-5 )



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
    
    f = wrapper.wrap_objective_function(f)

    L = max(np.linalg.eigvals(hessian))

    lmo = frankwolfe.ProbabilitySimplexOracle(1.0)
    x0 = frankwolfe.compute_extreme_point(lmo, np.zeros(n))
    
    x00 = copy.deepcopy(x0)

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

    x00 = copy.deepcopy(x0)
    
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

    x00 = copy.deepcopy(x0)
    
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
    
    n = int(1e2)
    k = int(1e4)


    xpi = np.random.rand(n)
    total = sum(xpi)
    xp = xpi

    def f(x):
        return np.linalg.norm(x - xp)**2
    
    def grad(storage, x):
        storage[:] = 2 * (x - xp)

    f = wrapper.wrap_objective_function(f)

    lmo = frankwolfe.KSparseLMO(40, 1.0)

    x0 = frankwolfe.compute_extreme_point(lmo, np.zeros(n))
    
    x00 = copy.deepcopy(x0)

    x, v, primal, dual_gap, trajectory = frankwolfe.frank_wolfe(
        f,
        grad,
        lmo,
        x00,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=100.0),
        print_iter=k / 10,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        epsilon=1e-5,
        trajectory=True,
    )

    x00 = copy.deepcopy(x0)

    x, v, primal, dual_gap, trajectory, active_set = frankwolfe.away_frank_wolfe(
        f,
        grad,
        lmo,
        x00,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=100.0),
        print_iter=k / 10,
        epsilon=1e-5,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        trajectory=True,
        lazy=True,
    )

    x00 = copy.deepcopy(x0)

    x, v, primal, dual_gap, trajectory_away, active_set = frankwolfe.away_frank_wolfe(
        f,
        grad,
        lmo,
        x00,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=100.0),
        print_iter=k / 10,
        epsilon=1e-5,
        memory_mode=frankwolfe.InplaceEmphasis(),
        verbose=True,
        trajectory=True,
        away_steps = True,
    )

    x00 = copy.deepcopy(x0)

    x, v, primal, dual_gap, trajectory_away_outplace , active_set = frankwolfe.away_frank_wolfe(
        f,
        grad,
        lmo,
        x00,
        max_iteration=k,
        line_search=frankwolfe.Adaptive(L_est=100.0),
        print_iter=k / 10,
        epsilon=1e-5,
        momentum=0.9,
        memory_mode=frankwolfe.OutplaceEmphasis(),
        verbose=True,
        trajectory=True,
        away_steps=True,
    )

