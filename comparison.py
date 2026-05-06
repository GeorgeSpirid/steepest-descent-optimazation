import numpy as np

from functions import *
from methods import *

def run_statistics(name, method_runner, n_trials, get_start):
    f_vals = []
    iters_vals = []
    
    for _ in range(n_trials):
        start = get_start()
        try:
            _, f_opt, iters = method_runner(start)
            if np.isnan(f_opt) or np.isinf(f_opt):
                continue
            f_vals.append(f_opt)
            iters_vals.append(iters)
        except Exception:
            continue
            
    if len(f_vals) == 0:
        print(f"[{name:18s}] Overflow in all trials")
        return

    mean_f = np.mean(f_vals)
    std_f = np.std(f_vals)
    mean_iters = np.mean(iters_vals)
    std_iters = np.std(iters_vals)
    
    print(f"[{name:18s}] f(x): Mean = {mean_f:12.6f} ± {std_f:9.6f} | Iters: Mean = {mean_iters:6.1f} ± {std_iters:5.1f}")


if __name__ == "__main__":
    N_TRIALS = 30 
    print("="*80)
    print(f" ΕΚΤΕΛΕΣΗ ΣΤΑΤΙΣΤΙΚΩΝ ΠΕΙΡΑΜΑΤΩΝ (N_TRIALS = {N_TRIALS})")
    print("="*80)

    # =====================================================================
    # 1. SPHERE FUNCTION (Convex) | Dimensions: n = 15
    # =====================================================================
    print("\n" + "-"*70)
    print(" 1. SPHERE FUNCTION (Convex) | Dimensions: n = 15")
    print("-"*70)
    def get_start_sphere(): return np.random.uniform(-5, 5, 15)
    
    run_statistics("Steepest Descent", lambda x: steepest_descent(x, sphere, sphere_gradient, sphere_domain), N_TRIALS, get_start_sphere)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, sphere, sphere_gradient, alpha=0.01, beta=0.9), N_TRIALS, get_start_sphere)
    run_statistics("NAG",              lambda x: nag(x, sphere, sphere_gradient, alpha=0.01, beta=0.9), N_TRIALS, get_start_sphere)
    run_statistics("BB1",              lambda x: bb(x, sphere, sphere_gradient, sphere_domain, variant='BB1'), N_TRIALS, get_start_sphere)


    # =====================================================================
    # 2. RASTRIGIN FUNCTION (Non-Convex) | Dimensions: n = 15
    # =====================================================================
    print("\n" + "-"*70)
    print(" 2. RASTRIGIN FUNCTION (Non-Convex) | Dimensions: n = 15")
    print("-"*70)
    # Ξεκινάμε από ένα πιο στενό πεδίο για να προλάβουμε άμεσες αποκλίσεις
    def get_start_rastrigin(): return np.random.uniform(-1, 1, 15)
    
    run_statistics("Steepest Descent", lambda x: steepest_descent(x, rastrigin, rastrigin_gradient, rastrigin_domain), N_TRIALS, get_start_rastrigin)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, rastrigin, rastrigin_gradient, alpha=0.001, beta=0.9), N_TRIALS, get_start_rastrigin)
    run_statistics("NAG",              lambda x: nag(x, rastrigin, rastrigin_gradient, alpha=0.001, beta=0.9), N_TRIALS, get_start_rastrigin)
    run_statistics("BB1",              lambda x: bb(x, rastrigin, rastrigin_gradient, rastrigin_domain, variant='BB1'), N_TRIALS, get_start_rastrigin)


    # =====================================================================
    # 3. ROSENBROCK FUNCTION (Non-Convex) | Dimensions: n = 15
    # =====================================================================
    print("\n" + "-"*70)
    print(" 3. ROSENBROCK FUNCTION (Non-Convex) | Dimensions: n = 15")
    print("-"*70)
    def get_start_rosenbrock(): return np.random.uniform(-1.5, 1.5, 15)
    
    run_statistics("Steepest Descent", lambda x: steepest_descent(x, rosenbrock, rosenbrock_gradient, rosenbrock_domain), N_TRIALS, get_start_rosenbrock)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, rosenbrock, rosenbrock_gradient, alpha=1e-4, beta=0.9), N_TRIALS, get_start_rosenbrock)
    run_statistics("NAG",              lambda x: nag(x, rosenbrock, rosenbrock_gradient, alpha=1e-4, beta=0.9), N_TRIALS, get_start_rosenbrock)
    run_statistics("BB1",              lambda x: bb(x, rosenbrock, rosenbrock_gradient, rosenbrock_domain, variant='BB1'), N_TRIALS, get_start_rosenbrock)


    # =====================================================================
    # 4. ACKLEY FUNCTION (Non-Convex) | Dimensions: n = 2
    # =====================================================================
    print("\n" + "-"*70)
    print(" 4. ACKLEY FUNCTION (Non-Convex) | Dimensions: n = 2")
    print("-"*70)
    def get_start_ackley(): return np.random.uniform(-3, 3, 2)
    
    run_statistics("Steepest Descent", lambda x: steepest_descent(x, ackley, ackley_gradient, ackley_domain), N_TRIALS, get_start_ackley)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, ackley, ackley_gradient, alpha=0.005, beta=0.9), N_TRIALS, get_start_ackley)
    run_statistics("NAG",              lambda x: nag(x, ackley, ackley_gradient, alpha=0.005, beta=0.9), N_TRIALS, get_start_ackley)
    run_statistics("BB1",              lambda x: bb(x, ackley, ackley_gradient, ackley_domain, variant='BB1'), N_TRIALS, get_start_ackley)


    # =====================================================================
    # 5. QUADRATIC FUNCTION (Convex) | Dimensions: n = 2
    # =====================================================================
    print("\n" + "-"*70)
    print(" 5. QUADRATIC FUNCTION (Convex) | Dimensions: n = 2")
    print("    [Το Ολικό Ελάχιστο f(x*) είναι στο -0.5]")
    print("-"*70)
    def get_start_quad(): return np.random.uniform(-5, 5, 2)
    
    # Υπολογισμός των βέλτιστων θεωρητικών παραμέτρων
    eigenvalues = np.linalg.eigvalsh(Q_mat)
    l1, ln = np.min(eigenvalues), np.max(eigenvalues)
    kappa = ln / l1
    alpha_hb_opt = 4 / (np.sqrt(ln) + np.sqrt(l1))**2
    beta_hb_opt = (np.sqrt(ln) - np.sqrt(l1))**2 / (np.sqrt(ln) + np.sqrt(l1))**2
    
    run_statistics("Steepest Descent", lambda x: steepest_descent_quadratic(x, quadratic, quadratic_gradient), N_TRIALS, get_start_quad)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, quadratic, quadratic_gradient, alpha=alpha_hb_opt, beta=beta_hb_opt), N_TRIALS, get_start_quad)
    run_statistics("NAG",              lambda x: nag(x, quadratic, quadratic_gradient, alpha=1/(ln+l1), beta=1/kappa), N_TRIALS, get_start_quad)
    run_statistics("BB1",              lambda x: bb(x, quadratic, quadratic_gradient, (-np.inf, np.inf), variant='BB1'), N_TRIALS, get_start_quad)


    # =====================================================================
    # 6. BOOTH FUNCTION (Convex) | Dimensions: n = 2
    # =====================================================================
    print("\n" + "-"*70)
    print(" 6. BOOTH FUNCTION (Convex) | Dimensions: n = 2")
    print("-"*70)
    def get_start_booth(): return np.random.uniform(-5, 5, 2)
    
    run_statistics("Steepest Descent", lambda x: steepest_descent(x, booth, booth_gradient, booth_domain), N_TRIALS, get_start_booth)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, booth, booth_gradient, alpha=0.01, beta=0.9), N_TRIALS, get_start_booth)
    run_statistics("NAG",              lambda x: nag(x, booth, booth_gradient, alpha=0.01, beta=0.9), N_TRIALS, get_start_booth)
    run_statistics("BB1",              lambda x: bb(x, booth, booth_gradient, booth_domain, variant='BB1'), N_TRIALS, get_start_booth)


    # =====================================================================
    # 7. MATYAS FUNCTION (Convex) | Dimensions: n = 2
    # =====================================================================
    print("\n" + "-"*70)
    print(" 7. MATYAS FUNCTION (Convex) | Dimensions: n = 2")
    print("-"*70)
    def get_start_matyas(): return np.random.uniform(-5, 5, 2)
    
    run_statistics("Steepest Descent", lambda x: steepest_descent(x, matyas, matyas_gradient, matyas_domain), N_TRIALS, get_start_matyas)
    run_statistics("Heavy Ball",       lambda x: heavy_ball(x, matyas, matyas_gradient, alpha=0.5, beta=0.9), N_TRIALS, get_start_matyas)
    run_statistics("NAG",              lambda x: nag(x, matyas, matyas_gradient, alpha=0.5, beta=0.9), N_TRIALS, get_start_matyas)
    run_statistics("BB1",              lambda x: bb(x, matyas, matyas_gradient, matyas_domain, variant='BB1'), N_TRIALS, get_start_matyas)
    
    print("\n" + "="*80)