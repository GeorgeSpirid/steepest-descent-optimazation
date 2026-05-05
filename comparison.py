import numpy as np
from scipy.optimize import minimize_scalar



# Rastrigin function, xi ε [-5.12, 5.12]
rastrigin_domain = (-5.12, 5.12)
def rastrigin(x): # f(0, ..., 0) = 0 is the global minimum
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_gradient(x):
    A = 10
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

ackley_domain = (-5, 5)
def ackley(x): # f(0, 0) = 0 is the global minimum
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = 2
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)
def ackley_gradient(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2) + 1e-12  # avoid division by zero
    sum2 = np.sum(np.cos(c * x))
    term1 = a * b * np.exp(-b * np.sqrt(sum1 / n)) * x / (n * np.sqrt(sum1 / n))
    term2 = c * np.exp(sum2 / n) * np.sin(c * x) / n
    return term1 + term2

sphere_domain = (-np.inf, np.inf)
def sphere(x): # f(0, ..., 0) = 0 is the global minimum
    return np.sum(x**2)
def sphere_gradient(x):
    return 2 * x

rosenbrock_domain = (-np.inf, np.inf)
def rosenbrock(x): # f(1, ..., 1) = 0 is the global minimum
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    for i in range(1, len(x)-1):
        grad[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return grad

Q_mat = np.array([[5.0, 1.0], [1.0, 2.0]])
b_vec = np.array([1.0, 1.0])

def quadratic(x):
    return 0.5 * x.T @ Q_mat @ x - b_vec.T @ x

def quadratic_gradient(x):
    return Q_mat @ x - b_vec

def max_a(x, g, domain):
    # largest step size a such that x-a*g is still in the domain
    lo, hi = domain
    bounds = []
    for xi, gi in zip(x, g):
        # x[i] - alpha*g[i] >= lo
        if gi > 0:
            bounds.append((xi - lo) / gi)
        # x[i] - alpha*g[i] <= hi
        elif gi < 0:
            bounds.append((hi - xi) / (-gi))
    return min(bounds) if bounds else 1.0

def line_search(x, g, func, domain):
    a_max = max_a(x, g, domain)
    a_max = max(a_max, 0) # non-negative step size
    if a_max <= 0:
        return 0.0
    phi = lambda a: func(x - a * g)
    if np.isinf(a_max):
        result = minimize_scalar(phi)
    else:
        result = minimize_scalar(phi, bounds=(0, a_max), method='bounded')
    return result.x

def line_search_quadratic(x, g):
    Qg = Q_mat @ g # alpha_k = <g_k, g_k> / <g_k, Q g_k>
    return (g @ g) / (g @ Qg)

def steepest_descent(x0, func, grad, domain, epsilon=1e-8, max_iterations=1000):
    """
    Finds a local minimum near the starting point
    Has no escape mechanism
    It may not find the global minimum
    """
    x = np.array(x0, dtype=float)
    k = 0
    while k < max_iterations:
        g = grad(x)
        if np.linalg.norm(g) < epsilon:
            break
        a_k = line_search(x, g, func, domain)
        x = x - a_k * g
        k += 1
    return x, func(x), k

def steepest_descent_quadratic(x0, func, grad, epsilon=1e-8, max_iterations=1000):
    x = np.array(x0, dtype=float)
    for k in range(max_iterations):
        g = quadratic_gradient(x)
        if np.linalg.norm(g) < epsilon: break
        a = line_search_quadratic(x, g)
        x = x - a * g
    return x, quadratic(x), k+1

def heavy_ball(x0, func, grad, alpha, beta, epsilon=1e-8, max_iterations=1000):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    k = 0
    while k < max_iterations:
        g = grad(x)
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)):
            break
        if np.linalg.norm(g) < epsilon:
            break
        v = beta*v + g # z^{k+1} = β·z^k + ∇f(w^k)
        x = x - alpha*v # w^{k+1} = w^k − α·z^{k+1}
        k += 1
    return x, func(x), k

def nag(x0, func, grad, alpha, beta, epsilon=1e-8, max_iterations=1000):
    w = np.array(x0, dtype=float)   # w^k
    w_prev = np.array(x0, dtype=float)   # w^{k-1} = w^0 at start
    k = 0
    while k < max_iterations:
        y = w + beta*(w - w_prev) # y^k = w^k + beta*(w^k - w^{k-1})
        g = grad(y) # grad_f(y^k)  <- lookahead gradient
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(w)): break
        if np.linalg.norm(g) < epsilon: break
        w_prev = w.copy()
        w = w - alpha*g # w^{k+1} = w^k - alpha*grad_f(y^k)
        k += 1
    return w, func(w), k+1

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
            alpha=1e-5, 
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
            start, func=sphere, grad=sphere_gradient, alpha=0.001, beta=0.9
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
    x_opt, f_opt, iterations = nag(
        x0_quad, func=quadratic, grad=quadratic_gradient, alpha=alpha_opt, beta=beta_nag
    )
    print(f"alpha*={alpha_opt:.6f}  beta*=1/kappa={beta_nag:.6f}  theoretical rate={rate_nag:.4f}")
    print(f"x0={list(x0_quad)} | x={np.round(x_opt,6)} | f={f_opt:.6f} | k={iterations}")