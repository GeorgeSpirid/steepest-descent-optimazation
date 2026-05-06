import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Εισαγωγή των συναρτήσεων και των μεθόδων γραμμικής αναζήτησης
from functions import *
from methods import line_search, line_search_quadratic

# =====================================================================
# ΣΥΝΑΡΤΗΣΕΙΣ ΚΑΤΑΓΡΑΦΗΣ ΜΟΝΟΠΑΤΙΟΥ (PATH COLLECTORS)
# Τρέχουν τους αλγόριθμους και αποθηκεύουν τα σημεία σε κάθε βήμα
# =====================================================================

def steepest_descent_path(x0, func, grad, domain, is_quad=False, max_iters=50):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iters):
        g = grad(x)
        if np.linalg.norm(g) < 1e-8: break
        if is_quad:
            a = line_search_quadratic(x, g)
        else:
            a = line_search(x, g, func, domain)
        x = x - a * g
        path.append(x.copy())
    return np.array(path)

def heavy_ball_path(x0, grad, alpha, beta, max_iters=50):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(max_iters):
        g = grad(x)
        if np.linalg.norm(g) < 1e-8: break
        v = beta * v + g
        x = x - alpha * v
        path.append(x.copy())
    return np.array(path)

def nag_path(x0, grad, alpha, beta, max_iters=50):
    w = np.array(x0, dtype=float)
    w_prev = np.array(x0, dtype=float)
    path = [w.copy()]
    for _ in range(max_iters):
        y = w + beta * (w - w_prev)
        g = grad(y)
        if np.linalg.norm(g) < 1e-8: break
        w_prev = w.copy()
        w = w - alpha * g
        path.append(w.copy())
    return np.array(path)

def bb1_path(x0, grad, max_iters=50):
    x = np.array(x0, dtype=float)
    g = grad(x)
    alpha = 0.001 
    x_prev = x.copy()
    g_prev = g.copy()
    
    x = x - alpha * g
    path = [x_prev.copy(), x.copy()]
    g = grad(x)
    
    for _ in range(1, max_iters):
        if np.linalg.norm(g) < 1e-8: break
        s = x - x_prev
        y = g - g_prev
        sy = s @ y
        if abs(sy) < 1e-14: break
        
        alpha = abs((s @ s) / sy)
        x_prev = x.copy()
        g_prev = g.copy()
        x = x - alpha * g
        g = grad(x)
        path.append(x.copy())
    return np.array(path)

# =====================================================================
# ΣΥΝΑΡΤΗΣΗ ΔΗΜΙΟΥΡΓΙΑΣ ΔΙΑΔΟΧΙΚΟΥ ANIMATION
# =====================================================================

def animate_function(title, X, Y, Z, paths_dict, global_min, xlim, ylim):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(X, Y, Z, levels=35, cmap='viridis', alpha=0.6)
    
    # Ζωγράφισε το Ολικό Ελάχιστο με ένα αστέρι
    ax.plot(global_min[0], global_min[1], 'r*', markersize=15, label='Global Min', zorder=5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    colors = ['red', 'blue', 'green', 'purple']
    lines = {}
    method_names = list(paths_dict.keys())
    
    # Αρχικοποίηση κενών γραμμών
    for name, color in zip(method_names, colors):
        line, = ax.plot([], [], color=color, marker='o', markersize=4, label=name, alpha=0.8, linewidth=1.5)
        lines[name] = line
        
    ax.legend(loc='upper right')
    
    # Υπολογισμός συνολικών frames (το άθροισμα των βημάτων όλων των μεθόδων)
    lengths = [len(paths_dict[name]) for name in method_names]
    total_frames = sum(lengths)
    
    def update(frame):
        current_offset = 0
        for name in method_names:
            path = paths_dict[name]
            L = len(path)
            
            if frame < current_offset:
                # Δεν έχει έρθει ακόμα η σειρά αυτής της μεθόδου
                lines[name].set_data([], [])
            elif frame >= current_offset + L:
                # Έχει ολοκληρωθεί, άστην πλήρως σχεδιασμένη
                lines[name].set_data(path[:, 0], path[:, 1])
            else:
                # Ζωγραφίζεται ΤΩΡΑ (frame by frame)
                idx = frame - current_offset
                lines[name].set_data(path[:idx+1, 0], path[:idx+1, 1])
                
            current_offset += L
            
        return list(lines.values())
        
    # Το repeat=False σταματάει το animation όταν ζωγραφιστούν όλα
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True, repeat=False)
    
    print(f"\nΕμφάνιση παραθύρου: {title}")
    print("Κλείσε το παράθυρο για να προχωρήσεις στο επόμενο!")
    plt.tight_layout()
    plt.show()

# =====================================================================
# ΠΑΡΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ ΚΑΙ ΕΚΤΕΛΕΣΗ
# =====================================================================

if __name__ == "__main__":
    
    # Κοινές παράμετροι πλέγματος
    x_grid = np.linspace(-5.12, 5.12, 200)
    y_grid = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    print("Υπολογισμός μονοπατιών...")
    
    # ---------------------------------------------------------
    # 1. QUADRATIC DATA
    # ---------------------------------------------------------
    start_quad = np.array([4.0, 4.0])
    eigenvalues = np.linalg.eigvalsh(Q_mat)
    l1, ln = np.min(eigenvalues), np.max(eigenvalues)
    kappa = ln / l1

    paths_quad = {
        'Steepest Descent': steepest_descent_path(start_quad, quadratic, quadratic_gradient, (-10,10), is_quad=True, max_iters=50),
        'Heavy Ball': heavy_ball_path(start_quad, quadratic_gradient, alpha=4/(np.sqrt(ln)+np.sqrt(l1))**2, beta=(np.sqrt(ln)-np.sqrt(l1))**2/(np.sqrt(ln)+np.sqrt(l1))**2, max_iters=50),
        'NAG': nag_path(start_quad, quadratic_gradient, alpha=1/(ln+l1), beta=1/kappa, max_iters=50),
        'BB1': bb1_path(start_quad, quadratic_gradient, max_iters=50)
    }

    Z_quad = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_quad[i,j] = quadratic(np.array([X[i,j], Y[i,j]]))
            
    star_quad = np.linalg.solve(Q_mat, b_vec)
    
    # ---------------------------------------------------------
    # 2. RASTRIGIN DATA
    # ---------------------------------------------------------
    start_rast = np.array([3.5, 4.0])
    
    paths_rast = {
        'Steepest Descent': steepest_descent_path(start_rast, rastrigin, rastrigin_gradient, rastrigin_domain, is_quad=False, max_iters=50),
        'Heavy Ball': heavy_ball_path(start_rast, rastrigin_gradient, alpha=0.001, beta=0.9, max_iters=50),
        'NAG': nag_path(start_rast, rastrigin_gradient, alpha=0.001, beta=0.9, max_iters=50),
        'BB1': bb1_path(start_rast, rastrigin_gradient, max_iters=50)
    }

    Z_rast = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_rast[i,j] = rastrigin(np.array([X[i,j], Y[i,j]]))

    # =====================================================================
    # ANIMATION RUNNER
    # Τρέχει τη μία συνάρτηση και όταν κλείσεις το παράθυρο, τρέχει την άλλη
    # =====================================================================

    animate_function(
        title="Quadratic Function (Convex) - Sequential Methods", 
        X=X, Y=Y, Z=Z_quad, 
        paths_dict=paths_quad, 
        global_min=star_quad, 
        xlim=[-1, 5], ylim=[-1, 5]
    )

    animate_function(
        title="Rastrigin Function (Non-Convex) - Sequential Methods", 
        X=X, Y=Y, Z=Z_rast, 
        paths_dict=paths_rast, 
        global_min=np.array([0, 0]), 
        xlim=[-5.12, 5.12], ylim=[-5.12, 5.12]
    )