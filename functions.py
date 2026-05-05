import numpy as np

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

Q_mat = np.array([[100.0, 1.0], [1.0,   1.0]])
b_vec = np.array([1.0, 1.0])

def quadratic(x):
    return 0.5 * x.T @ Q_mat @ x - b_vec.T @ x

def quadratic_gradient(x):
    return Q_mat @ x - b_vec