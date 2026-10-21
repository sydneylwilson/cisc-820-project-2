import numpy as np

def forward_difference(func, x, h, order):
    """
    x: point (numpy array of size `order`)
    h: step size
    order: dimension of x
    """
    
    x = np.array(x, dtype=float)  # ensure we have a copy
    f0 = func(x)
    matrix = np.zeros(order)
    for i in range(order):
        tempx = x.copy()          
        tempx[i] += h
        matrix[i] = func(tempx)

    return (matrix - f0) / h


def central_difference(func, x, h, order):
    x = np.array(x, dtype=float)
    fmatrix = np.zeros(order)
    bmatrix = np.zeros(order)
    for i in range(order):
        ftempx = x.copy()
        btempx = x.copy()
        ftempx[i] += h
        btempx[i] -= h
        fmatrix[i] = func(ftempx)
        bmatrix[i] = func(btempx)   
    return (fmatrix - bmatrix) / (2 * h)

def hessian_forward(func, x, h, order):
    """
    Forward-difference approximation of the Hessian.
    H[i, j] ≈ (g_i(x + h*e_j) - g_i(x)) / h
    """
    x = np.array(x, dtype=float)
    H = np.zeros((order, order))

    g0 = forward_difference(func, x, h, order)   # baseline gradient

    for j in range(order):
        xtemp = x.copy()
        xtemp[j] += h
        gj = forward_difference(func, xtemp, h, order)
        H[:, j] = (gj - g0) / h

    return H

def hessian_central(func, x, h, order):
    """
    Central-difference approximation of the Hessian.
    H[i, j] ≈ (g_i(x + h*e_j) - g_i(x - h*e_j)) / (2h)
    """
    x = np.array(x, dtype=float)
    H = np.zeros((order, order))

    for j in range(order):
        xf = x.copy(); xf[j] += h
        xb = x.copy(); xb[j] -= h

        gf = central_difference(func, xf, h, order)
        gb = central_difference(func, xb, h, order)
        H[:, j] = (gf - gb) / (2 * h)

    return H

def func1(x):
    return x[0]**2 + 10*x[1]**2  # f(x,y,z) = x²+y²+z²
    # gradient = [2x_1, 20x_2]

# h = 1e-5
# order = 3

# print("Forward diff:", forward_difference(func, x, h, order))
# print("Central diff:", central_difference(func, x, h, order))
# print("Exact grad:  ", 2 * x)



# Backtracking line search (Armijo sufficient decrease)
def back_tracking(func, x, c, rho, alpha_bar, grad_func=None, h_grad=1e-5):
    """
    func      : objective function f(x)
    x         : starting point (numpy array)
    c         : Armijo constant (e.g. 1e-4)
    rho       : reduction factor in (0, 1)
    alpha_bar : initial trial step size
    grad_func : optional gradient function; if None, uses central_difference
    h_grad    : step size for the finite-difference gradient

    Returns
    -------
    alpha  : accepted step size
    p      : descent direction used
    trials : number of trials
    alphas : list of trial step sizes
    """
    x = np.array(x, dtype=float)
    n = x.size

    # --- 1. descent direction: p = -grad f(x) ---
    if grad_func is None:
        g = central_difference(func, x, h_grad, n)
    else:
        g = grad_func(x)
    p = -g

    # --- 2. phi(0) and phi'(0)
    phi0       = func(x)
    phi0_prime = g @ p          

    # --- 3. backtracking loop ---
    alpha  = alpha_bar
    trials = 0
    alphas = []

    while True:
        trials += 1
        alphas.append(alpha)

        phi_alpha  = func(x + alpha * p)
        armijo_rhs = phi0 + c * alpha * phi0_prime

        if phi_alpha <= armijo_rhs:
            break

        alpha *= rho

    return alpha, p, trials, alphas


def gradient_descent(func, x0, c=1e-4, rho=0.5, alpha_bar=1.0,
                     grad_func=None, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        alpha, p, trials, alphas = back_tracking(
            func, x, c, rho, alpha_bar, grad_func
        )

        x = x + alpha * p

        # convergence check 
        if np.linalg.norm(p) < tol:   
            break

    return x

from numpy.linalg import inv
from scipy.linalg import solve

def newton_direction(func, x, c, rho, alpha_bar,
                     grad_func=None, h_grad=1e-5, max_trials=100):
    x = np.array(x, dtype=float)
    n = x.size

    #  1. gradient and Hessian 
    if grad_func is None:
        g = central_difference(func, x, h_grad, n)
    else:
        g = grad_func(x)
    H = hessian_central(func, x, h_grad, n)

    # 2. Newton direction: solve H p = -g 
    try:
        p = solve(H, -g, assume_a='pos')   
    except np.linalg.LinAlgError:
        p = -g                              # fall back to gradient descent

    #  3. default to gradient-descent safeguard 
    if g @ p >= 0:
        p = -g                              

    #  4. Armijo backtracking 
    phi0       = func(x)
    phi0_prime = g @ p                      

    alpha  = alpha_bar
    trials = 0
    alphas = []

    while trials < max_trials:
        trials += 1
        alphas.append(alpha)

        phi_alpha  = func(x + alpha * p)
        armijo_rhs = phi0 + c * alpha * phi0_prime

        if phi_alpha <= armijo_rhs:
            break

        alpha *= rho
        if alpha < 1e-16:
            break

    return alpha, p, trials, alphas



def newton_optimize(func, x0, grad_func=None, hess_func=None,
                    c=0.0001, rho=0.5, alpha_bar=1.0,
                    tol=1e-6, max_iter=20, verbose=True):
    """
    Full Newton method with backtracking line search.
    Calls newton_direction at each iteration.
    """
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        alpha, p, trials, alphas = newton_direction(
            func, x, c, rho, alpha_bar,
            grad_func=grad_func, max_trials=100
        )

        # gradient norm for convergence check
        g = grad_func(x) if grad_func is not None else central_difference(func, x, 1e-5, len(x))
        gnorm = np.linalg.norm(g)

        if verbose:
            print(f"iter {k:3d} | f = {func(x):.6e} | ||g|| = {gnorm:.3e} "
                  f"| alpha = {alpha:.3e} | trials = {trials}")

        # convergence check
        if gnorm < tol:
            if verbose:
                print(f"  → converged at iteration {k}")
            break

        # UPDATE
        x = x + alpha * p

    return x, k
# print(back_tracking(func1, [1,1], 1e-4, 0.5, 1.0))

def f4(x):
    return (x[0]-2)**2 + 10*(x[1]-1)**2 + np.exp(x[0]*x[1])


def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + 10*x[1]**2

def f3(x):
    return (np.exp(x[0] + 3*x[1] - 0.1)
            + np.exp(x[0] - 3*x[1] - 0.1)
            + np.exp(-x[0] - 0.1))

print("=" * 60)
print("Test 1: f1 = x1^2 + x2^2,  start (50, 50)")
print("=" * 60)
x1_final, it1 = newton_optimize(f1, [50.0, 50.0])
print(f"Final x = {x1_final}, f = {f1(x1_final):.3e}\n")

# print("=" * 60)
# print("Test 2: f2 = x1^2 + 10*x2^2,  start (50, 50)")
# print("=" * 60)
# x2_final, it2 = newton_optimize(f2, [50.0, 50.0])
# print(f"Final x = {x2_final}, f = {f2(x2_final):.3e}\n")

# print("=" * 60)
# print("Test 3: f3 = exponential,  start (2, 1)")
# print("=" * 60)
# x3_final, it3 = newton_optimize(f3, [2.0, 1.0])
# print(f"Final x = {x3_final}, f = {f3(x3_final):.3e}\n")

print("=" * 60)
print("Test 4: f4 = quadratic + exponential,  start (1, 1)")
print("=" * 60)
x4_final, it4 = newton_optimize(f4, [1.0, 1.0])
print(f"Final x = {x4_final}, f = {f4(x4_final):.3e}\n")

print("=" * 60)
print("Test 5: f4 = quadratic + exponential,  start (2, 2)")
print("=" * 60)
x5_final, it5 = newton_optimize(f4, [2.0, 2.0])
print(f"Final x = {x5_final}, f = {f4(x5_final):.3e}\n")

print("=" * 60)
print("Test 6: f4 = quadratic + exponential,  start (-2, -2)")
print("=" * 60)
x6_final, it6 = newton_optimize(f4, [-2.0, -2.0])
print(f"Final x = {x6_final}, f = {f4(x6_final):.3e}\n")

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"f1: {it1} iterations")
# print(f"f2: {it2} iterations")
# print(f"f3: {it3} iterations")
print(f"f4: {it4} iterations")
print(f"f5: {it5} iterations")
print(f"f6: {it6} iterations")