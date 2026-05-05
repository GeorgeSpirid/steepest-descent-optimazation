import numpy as np
from scipy.optimize import minimize_scalar

from functions import *

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
        y = np.clip(y, -1e6, 1e6) # stop overflow
        g = grad(y) # grad_f(y^k)  <- lookahead gradient
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(w)): break
        if np.linalg.norm(g) < epsilon: break
        w_prev = w.copy()
        w = w - alpha*g # w^{k+1} = w^k - alpha*grad_f(y^k)
        k += 1
    return w, func(w), k+1

def bb(x0, func, grad, domain, variant='BB1', epsilon=1e-8, max_iterations=1000):
    x = np.array(x0, dtype=float)
    g = grad(x)
    # line search to geta  starting alpha
    a_max = max_a(x, g, domain)
    if np.isinf(a_max):
        phi = lambda a: func(x - a*g)
        result = minimize_scalar(phi, method='brent')
        alpha = result.x
    else:
        phi = lambda a: func(x - a*g)
        result = minimize_scalar(phi, bounds=(0, a_max), method='bounded')
        alpha = result.x

    x_prev = x.copy()
    g_prev = g.copy()
    x = x - alpha*g
    g = grad(x)

    for k in range(1, max_iterations):
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)): break
        if np.linalg.norm(g) < epsilon: break

        s  = x - x_prev # s = x^k - x^{k-1}
        y  = g - g_prev # y = g^k - g^{k-1}
        sy = s @ y

        if abs(sy) < 1e-14: # avoid division by zero
            break

        if variant == 'BB1':
            alpha = (s@s) / sy # BB1: <s,s>/<s,y>
        else:
            alpha = sy / (y@y) # BB2: <s,y>/<y,y>

        alpha = abs(alpha) # ensure positive step size

        x_prev = x.copy()
        g_prev = g.copy()
        x = x - alpha*g # x^{k+1} = x^k - alpha_k * g^k
        g = grad(x)

    return x, func(x), k+1