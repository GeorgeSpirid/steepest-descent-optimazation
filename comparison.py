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

def steepest_descent(x0, func, grad, domain, epsilon=1e-8, max_iterations=10000):
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
    good_rastrigin_start = np.array([0.3, -0.2])
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