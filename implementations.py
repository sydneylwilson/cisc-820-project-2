# project2.py
import numpy as np
from functions import gradient_descent, newton_optimize, bfgs_optimize


W1 = np.arange(1, 101, dtype=float)
def f1(x):       return np.sum(W1 * x * x)
def grad_f1(x):  return 2.0 * W1 * x
def hess_f1(x):  return np.diag(2.0 * W1)


A2 = np.loadtxt("fun2_A.txt").reshape((500, 100), order="F")
b2 = np.loadtxt("fun2_b.txt")
c2 = np.loadtxt("fun2_c.txt")

def f2(x):
    r = b2 - A2 @ x
    if np.any(r <= 0):
        return np.inf
    return c2 @ x - np.sum(np.log(r))

def grad_f2(x):
    r = b2 - A2 @ x
    return c2 + A2.T @ (1.0 / r)

def hess_f2(x):
    r = b2 - A2 @ x
    d = 1.0 / (r * r)
    # return A2.T @ (d[:, None] * A2)
    return A2.T @ np.diag(d) @ A2

def f3(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def grad_f3(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
                      200*(x[1]-x[0]**2)])
def hess_f3(x):
    return np.array([[-400*(x[1]-x[0]**2) + 800*x[0]**2 + 2, -400*x[0]],
                     [-400*x[0], 200]])


def run_all(name, f, g, H, x0, max_iter_gd=10000, max_iter_nt=200):
    print(f"\n {name} ")
    # print(f"  x0 = {x0}")

    x_gd, k_gd = gradient_descent(f, x0, grad_func=g,
                                  tol=1e-10, max_iter=max_iter_gd, verbose=False)
    print(f"  GD   : f = {f(x_gd):.6e} | ||g|| = {np.linalg.norm(g(x_gd)):.2e} | iters = {k_gd}")

    x_nt, k_nt = newton_optimize(f, x0, grad_func=g, hess_func=H,
                                 tol=1e-10, max_iter=max_iter_nt, verbose=False)
    print(f"  NT   : f = {f(x_nt):.6e} | ||g|| = {np.linalg.norm(g(x_nt)):.2e} | iters = {k_nt}")

    x_bf, k_bf = bfgs_optimize(f, x0, grad_func=g,
                               tol=1e-10, max_iter=max_iter_gd, verbose=False)
    print(f"  BFGS : f = {f(x_bf):.6e} | ||g|| = {np.linalg.norm(g(x_bf)):.2e} | iters = {k_bf}")

    return (x_gd, k_gd), (x_nt, k_nt), (x_bf, k_bf)

#  Function 1 
run_all("F1", f1, grad_f1, hess_f1,
        x0=np.ones(100) * 2.0)

#  Function 2   
x0_2 = np.zeros(100) 
run_all("F2", f2, grad_f2, hess_f2, x0=x0_2,
        max_iter_gd=5000, max_iter_nt=100)

#  Function 3   
run_all("F3", f3, grad_f3, hess_f3, x0=np.array([-1.2, 1.0]))