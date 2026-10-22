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

def back_tracking(func, x, c, rho, alpha_bar, grad_func=None, h_grad=1e-5,
                  direction=None, grad=None, max_trials=100):
    x = np.array(x, dtype=float)
    n = x.size

    # --- 1. gradient (reuse if provided) ---
    if grad is None:
        if grad_func is None:
            g = central_difference(func, x, h_grad, n)
        else:
            g = grad_func(x)
    else:
        g = grad

    p = -g if direction is None else np.array(direction, dtype=float)

    if g @ p >= 0:
        p = -g

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

    return alpha, p, g, trials, alphas

def gradient_descent(func, x0, c=1e-4, rho=0.5, alpha_bar=1.0,
                     grad_func=None, h_grad=1e-5,
                     tol=1e-12, max_iter=1000, verbose=True):
    """
    Gradient descent with backtracking line search (Armijo).
    Reuses back_tracking for the line search; direction is p = -g.
    """
    x = np.array(x0, dtype=float)
    n = x.size

    for k in range(max_iter):
        # gradient at current x (recomputed each iteration — x changes)
        if grad_func is None:
            g = central_difference(func, x, h_grad, n)
        else:
            g = grad_func(x)
        gnorm = np.linalg.norm(g)
        if verbose:
            print(f"iter {k:3d} | f = {func(x):.6e} | ||g|| = {gnorm:.3e}")

        if gnorm < tol:
            if verbose:
                print(f"  → converged at iteration {k}")
            break   

        # line search with steepest descent direction
        alpha, p, g, trials, alphas = back_tracking(
            func, x, c, rho, alpha_bar,
            grad_func=grad_func, h_grad=h_grad,
            direction=None,        # None → p = -g
            grad=g
        )

        

        # UPDATE
        x = x + alpha * p

    return x, k

from scipy.linalg import solve

def newton_optimize(func, x0, grad_func=None, hess_func=None,
                    c=1e-4, rho=0.5, alpha_bar=1.0, h_grad=1e-5,
                    tol=1e-12, max_iter=100, verbose=True):
    """
    Newton method with backtracking line search (Armijo).
    Reuses back_tracking for the line search; direction is p = -H^{-1} g.
    """
    x = np.array(x0, dtype=float)
    n = x.size

    for k in range(max_iter):
        # gradient at current x
        if grad_func is None:
            g = central_difference(func, x, h_grad, n)
        else:
            g = grad_func(x)

        gnorm = np.linalg.norm(g)

        if verbose:
            print(f"iter {k:3d} | f = {func(x):.6e} | ||g|| = {gnorm:.3e}")

        if gnorm < tol:
            if verbose:
                print(f"  → converged at iteration {k}")
            break

        # Hessian at current x (exact or finite-difference)
        if hess_func is None:
            H = hessian_central(func, x, h_grad, n)
        else:
            H = hess_func(x)

        # Newton direction: solve H p = -g
        try:
            p_newton = solve(H, -g, assume_a='pos')
        except np.linalg.LinAlgError:
            p_newton = -g                  # fall back to steepest descent

        # line search; back_tracking handles the descent safeguard
        alpha, p, g, trials, alphas = back_tracking(
            func, x, c, rho, alpha_bar,
            grad_func=grad_func, h_grad=h_grad,
            direction=p_newton,
            grad=g
        )

        if verbose:
            print(f"          alpha = {alpha:.3e} | trials = {trials}")

        # UPDATE
        x = x + alpha * p

    return x, k

def bfgs_optimize(func, x0, grad_func=None, h_grad=1e-5,
                  c=1e-4, rho=0.5, alpha_bar=1.0,
                  tol=1e-12, max_iter=1000, verbose=True):
    x = np.array(x0, dtype=float)
    n = x.size
    C = np.eye(n)

    for k in range(max_iter):
        # gradient at current x
        if grad_func is None:
            g = central_difference(func, x, h_grad, n)
        else:
            g = grad_func(x)

        gnorm = np.linalg.norm(g)

        if verbose:
            print(f"iter {k:3d} | f = {func(x):.6e} | ||g|| = {gnorm:.3e}")

        if gnorm < tol:
            if verbose:
                print(f"  → converged at iteration {k}")
            break

        # BFGS direction
        p_bfgs = -C @ g

        # line search
        alpha, p, g, trials, alphas = back_tracking(
            func, x, c, rho, alpha_bar,
            grad_func=grad_func, h_grad=h_grad,
            direction=p_bfgs,
            grad=g
        )

        if verbose:
            print(f"          alpha = {alpha:.3e} | trials = {trials}")

        # UPDATE position
        x_new = x + alpha * p

        # new gradient for secant pair
        if grad_func is None:
            g_new = central_difference(func, x_new, h_grad, n)
        else:
            g_new = grad_func(x_new)

        s = x_new - x
        y = g_new - g
        C = bfgs_update(C, s, y)

        x = x_new

    return x, k

def bfgs_update(C, s, y):
    """
    BFGS inverse update (factored form):

        C⁺ = (I - ρ s yᵀ) C (I - ρ y sᵀ) + ρ s sᵀ,   ρ = 1/(yᵀs)

    Returns the updated C. If the curvature condition yᵀs > 0
    fails, returns C unchanged (skips the update).
    """
    ys = y @ s
    if ys <= 1e-14:
        return C                     # curvature condition failed → skip

    rho = 1.0 / ys
    n = C.shape[0]
    I = np.eye(n)

    # V = I - ρ s yᵀ   (n x n)
    V = I - rho * np.outer(s, y)

    return V @ C @ V.T + rho * np.outer(s, s)



# def f4(x):
#     return (x[0]-2)**2 + 10*(x[1]-1)**2 + np.exp(x[0]*x[1])


# def f1(x):
#     return x[0]**2 + x[1]**2

# def f2(x):
#     return x[0]**2 + 10*x[1]**2

# def f3(x):
#     return (np.exp(x[0] + 3*x[1] - 0.1)
#             + np.exp(x[0] - 3*x[1] - 0.1)
#             + np.exp(-x[0] - 0.1))

# print("=" * 60)
# print("Test 1: f1 = x1^2 + x2^2,  start (50, 50)")
# print("=" * 60)
# x1_final, it1 = newton_optimize(f1, [50.0, 50.0])
# print(f"Final x = {x1_final}, f = {f1(x1_final):.3e}\n")

# print("=" * 60)
# print("Test 2: f1 = x1^2 + x2^2,  start (50, 50)")
# print("=" * 60)
# x2_final, it2 = gradient_descent(f1, [50.0, 50.0])
# print(f"Final x = {x2_final}, f = {f1(x2_final):.3e}\n")

# print("=" * 60)
# print("Test 3: f1 = x1^2 + x2^2,  start (50, 50)")
# print("=" * 60)
# x3_final, it3 = bfgs_optimize(f1, [50.0, 50.0])
# print(f"Final x = {x3_final}, f = {f1(x3_final):.3e}\n")

# # print("=" * 60)
# # print("Test 2: f2 = x1^2 + 10*x2^2,  start (50, 50)")
# # print("=" * 60)
# # x2_final, it2 = newton_optimize(f2, [50.0, 50.0])
# # print(f"Final x = {x2_final}, f = {f2(x2_final):.3e}\n")

# # print("=" * 60)
# # print("Test 3: f3 = exponential,  start (2, 1)")
# # print("=" * 60)
# # x3_final, it3 = newton_optimize(f3, [2.0, 1.0])
# # print(f"Final x = {x3_final}, f = {f3(x3_final):.3e}\n")

# print("=" * 60)
# print("Test 4: f4 = quadratic + exponential,  start (1, 1)")
# print("=" * 60)
# x4_final, it4 = newton_optimize(f4, [1.0, 1.0])
# print(f"Final x = {x4_final}, f = {f4(x4_final):.3e}\n")

# print("=" * 60)
# print("Test 5: f4 = quadratic + exponential,  start (2, 2)")
# print("=" * 60)
# x5_final, it5 = newton_optimize(f4, [2.0, 2.0])
# print(f"Final x = {x5_final}, f = {f4(x5_final):.3e}\n")

# print("=" * 60)
# print("Test 6: f4 = quadratic + exponential,  start (-2, -2)")
# print("=" * 60)
# x6_final, it6 = newton_optimize(f4, [-2.0, -2.0])
# print(f"Final x = {x6_final}, f = {f4(x6_final):.3e}\n")

# print("=" * 60)
# print("Test 7: f4 = quadratic + exponential,  start (-2, -2)")
# print("=" * 60)
# x7_final, it7 = bfgs_optimize(f4, [-2.0, -2.0])
# print(f"Final x = {x7_final}, f = {f4(x7_final):.3e}\n")

# print("=" * 60)
# print("Summary")
# print("=" * 60)
# print(f"f1: {it1} iterations")
# print(f"f2: {it2} iterations")
# print(f"f3: {it3} iterations")
# print(f"f4: {it4} iterations")
# print(f"f5: {it5} iterations")
# print(f"f6: {it6} iterations")
# print(f"f7: {it7} iterations")