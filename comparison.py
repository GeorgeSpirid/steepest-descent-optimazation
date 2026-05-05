import numpy as np

from functions import *
from methods import *

if __name__ == "__main__":
    n_dim = 2
    n_trials = 3

    print("=== Testing Rastrigin Function ===")
    starts_rastrigin = [np.random.uniform(*rastrigin_domain, n_dim) for _ in range(n_trials)]
    for start in starts_rastrigin:
        x_opt, f_opt, iterations = steepest_descent(
            start, 
            func=rastrigin, 
            grad=rastrigin_gradient, 
            domain=rastrigin_domain
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")
    good_rastrigin_start = np.array([0.1, 0.1])
    x_opt, f_opt, iterations = steepest_descent(
        good_rastrigin_start, 
        func=rastrigin, 
        grad=rastrigin_gradient, 
        domain=rastrigin_domain
    )
    print(f"Good Start: {np.round(good_rastrigin_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")

    print("\n=== Testing Ackley Function ===")
    starts_ackley = [np.random.uniform(*ackley_domain, n_dim) for _ in range(n_trials)]
    for start in starts_ackley:
        x_opt, f_opt, iterations = steepest_descent(
            start, 
            func=ackley, 
            grad=ackley_gradient, 
            domain=ackley_domain
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")
    good_ackley_start = np.array([-0.1, 0.4])
    x_opt, f_opt, iterations = steepest_descent(
        good_ackley_start, 
        func=ackley, 
        grad=ackley_gradient, 
        domain=ackley_domain
    )
    print(f"Good Start: {np.round(good_ackley_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")

    print("\n=== Testing Sphere Function ===")
    starts_sphere = [np.random.uniform(-10, 10, n_dim) for _ in range(n_trials)]
    for start in starts_sphere:
        x_opt, f_opt, iterations = steepest_descent(
            start, 
            func=sphere, 
            grad=sphere_gradient, 
            domain=sphere_domain
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")

    print("\n=== Testing Rosenbrock Function ===")
    starts_rosenbrock = [np.random.uniform(-2, 2, n_dim) for _ in range(n_trials)]
    for start in starts_rosenbrock:
        x_opt, f_opt, iterations = steepest_descent(
            start, 
            func=rosenbrock, 
            grad=rosenbrock_gradient, 
            domain=rosenbrock_domain
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")
    good_rosenbrock_start = np.array([1.2, 1.2])
    x_opt, f_opt, iterations = steepest_descent(
        good_rosenbrock_start, 
        func=rosenbrock, 
        grad=rosenbrock_gradient, 
        domain=rosenbrock_domain
    )
    print(f"Good Start: {np.round(good_rosenbrock_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")

    print("\n=== Testing Heavy Ball on Ackley Function ===")
    for start in starts_ackley:
        x_opt, f_opt, iterations = heavy_ball(
            start, 
            func=ackley, 
            grad=ackley_gradient, 
            alpha=0.001, 
            beta=0.9, 
            epsilon=1e-8, 
            max_iterations=1000
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")
    x_opt, f_opt, iterations = heavy_ball(
        good_ackley_start, 
        func=ackley, 
        grad=ackley_gradient, 
        alpha=0.001, 
        beta=0.9, 
        epsilon=1e-8, 
        max_iterations=1000
    )
    print(f"Good Start: {np.round(good_ackley_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")

    print("\n=== Testing Heavy Ball on Sphere Function ===")
    for start in starts_sphere:
        x_opt, f_opt, iterations = heavy_ball(
            start, 
            func=sphere, 
            grad=sphere_gradient, 
            alpha=0.001,
            beta=0.9, 
            epsilon=1e-8, 
            max_iterations=1000
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")
    
    print("\n=== Testing Heavy Ball on Rosenbrock Function ===")
    for start in starts_rosenbrock:
        x_opt, f_opt, iterations = heavy_ball(
            start, 
            func=rosenbrock, 
            grad=rosenbrock_gradient, 
            alpha=1e-5, 
            beta=0.9, 
            epsilon=1e-8, 
            max_iterations=1000
        )
        print(f"Start: {np.round(start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")
    x_opt, f_opt, iterations = heavy_ball(
        good_rosenbrock_start, 
        func=rosenbrock, 
        grad=rosenbrock_gradient, 
        alpha=1e-5, 
        beta=0.9, 
        epsilon=1e-8, 
        max_iterations=1000
    )
    print(f"Good Start: {np.round(good_rosenbrock_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations}")

    x_start = np.array([2.0, 2.0])
    x_star  = np.linalg.solve(Q_mat, b_vec) # x* = Q^{-1}b
    eigenvalues = np.linalg.eigvalsh(Q_mat)
    l1, ln = np.min(eigenvalues), np.max(eigenvalues)
    kappa = ln / l1
    alpha_opt = 4 / (np.sqrt(ln) + np.sqrt(l1))**2
    beta_opt = (np.sqrt(ln) - np.sqrt(l1))**2 / (np.sqrt(ln) + np.sqrt(l1))**2
    rate_opt = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    print("\n=== Testing Heavy Ball on Quadratic Function ===")
    x_opt, f_opt, iterations = heavy_ball(
        x_start, 
        func=quadratic, 
        grad=quadratic_gradient, 
        alpha=alpha_opt, 
        beta=beta_opt, 
        epsilon=1e-8, 
        max_iterations=1000
    )
    print(f"x* = {np.round(x_star, 4)}   f(x*) = {quadratic(x_star):.6f}   (global minimum)")
    print(f"Start: {np.round(x_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations} | Theoretical Rate: {rate_opt:.4f}")

    print("\n=== Testing Steepest Descent on Quadratic Function ===")
    alpha_sd = 2 / (ln + l1)
    rate_sd = (ln - l1) / (ln + l1)

    x_opt, f_opt, iterations = steepest_descent_quadratic(
        x_start, 
        func=quadratic, 
        grad=quadratic_gradient, 
        epsilon=1e-8, 
        max_iterations=1000
    )
    print(f"x* = {np.round(x_star, 4)}   f(x*) = {quadratic(x_star):.6f}   (global minimum)")

    print(f"Steepest Descent | Start: {np.round(x_start, 2)} | Optimal x: {np.round(x_opt, 4)} | f(x): {f_opt:.4f} | Iters: {iterations} | Theoretical Rate: {rate_sd:.4f}")
    print("\n=== NAG — Ackley (alpha=0.001, beta=0.9) ===")
    for start in starts_ackley:
        x_opt, f_opt, iterations = nag(
            start, func=ackley, grad=ackley_gradient, alpha=0.001, beta=0.9
        )
        print(f"Start: {np.round(start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")
    x_opt, f_opt, iterations = nag(
        good_ackley_start, func=ackley, grad=ackley_gradient, alpha=0.001, beta=0.9
    )
    print(f"Good Start: {np.round(good_ackley_start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")

    print("\n=== NAG — Sphere (alpha=0.001, beta=0.9) ===")
    for start in starts_sphere:
        x_opt, f_opt, iterations = nag(
            start, func=sphere, grad=sphere_gradient, alpha=0.0001, beta=0.9
        )
        print(f"Start: {np.round(start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")

    print("\n=== NAG — Rosenbrock (alpha=1e-5, beta=0.9) ===")
    for start in starts_rosenbrock:
        x_opt, f_opt, iterations = nag(
            start, func=rosenbrock, grad=rosenbrock_gradient, alpha=1e-5, beta=0.9
        )
        print(f"Start: {np.round(start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")
    x_opt, f_opt, iterations = nag(
        good_rosenbrock_start, func=rosenbrock, grad=rosenbrock_gradient, alpha=1e-5, beta=0.9
    )
    print(f"Good Start: {np.round(good_rosenbrock_start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")

    print("\n=== NAG — Quadratic (optimal alpha* and beta*=1/kappa) ===")
    # NAG optimal beta* = 1/kappa  (simpler than Heavy Ball's beta*)
    beta_nag  = 1 / kappa
    rate_nag  = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)   # same rate as HB
    x0_quad   = np.array([2.0, 2.0])
    alpha_nag = 1 / (ln + l1)
    x_opt, f_opt, iterations = nag(
        x0_quad, func=quadratic, grad=quadratic_gradient, alpha=alpha_nag, beta=beta_nag
    )
    print(f"x* = {np.round(x_star, 4)}   f(x*) = {quadratic(x_star):.6f}   (global minimum)")
    print(f"alpha*={alpha_nag:.6f}  beta*=1/kappa={beta_nag:.6f}  theoretical rate={rate_nag:.4f}")
    print(f"x0={list(x0_quad)} | x={np.round(x_opt,6)} | f={f_opt:.6f} | k={iterations}")

    print("\n=== BB1 — Ackley ===")
    for start in starts_ackley:
        x_opt, f_opt, iterations = bb(
            start, func=ackley, grad=ackley_gradient, domain=ackley_domain, variant='BB1'
        )
        print(f"Start: {np.round(start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")
    x_opt, f_opt, iterations = bb(
        good_ackley_start, func=ackley, grad=ackley_gradient, domain=ackley_domain, variant='BB1'
    )
    print(f"Good Start: {np.round(good_ackley_start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")

    print("\n=== BB1 — Sphere ===")
    for start in starts_sphere:
        x_opt, f_opt, iterations = bb(
            start, func=sphere, grad=sphere_gradient, domain=sphere_domain, variant='BB1'
        )
        print(f"Start: {np.round(start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")

    print("\n=== BB1 — Rosenbrock ===")
    for start in starts_rosenbrock:
        x_opt, f_opt, iterations = bb(
            start, func=rosenbrock, grad=rosenbrock_gradient, domain=rosenbrock_domain, variant='BB1'
        )
        print(f"Start: {np.round(start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")
    x_opt, f_opt, iterations = bb(
        good_rosenbrock_start, func=rosenbrock, grad=rosenbrock_gradient, domain=rosenbrock_domain, variant='BB1'
    )
    print(f"Good Start: {np.round(good_rosenbrock_start,2)} | x: {np.round(x_opt,4)} | f: {f_opt:.4f} | k: {iterations}")

    print("\n=== BB1 vs BB2 — Quadratic ===")
    print(f"x* = {np.round(x_star,4)}   f(x*) = {quadratic(x_star):.6f}   (global minimum)")
    for variant in ['BB1', 'BB2']:
        x_opt, f_opt, iterations = bb(
            x_start, func=quadratic, grad=quadratic_gradient, domain=(-np.inf, np.inf), variant=variant
        )
        print(f"{variant} | x={np.round(x_opt,6)} | f={f_opt:.6f} | k={iterations}")