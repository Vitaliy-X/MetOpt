import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.colors import Normalize
from sympy import Symbol, lambdify, pi, exp, sin, cos
from scipy.optimize import line_search


# Main.Task №1 (using numpy)
def newton_method(f, x, grad, hess, eps=1e-7,
                  method='standard', callback=None):
    """
    Perform Newton's method for optimization.

    Parameters:
    f (callable): Objective function.
    x (ndarray): Initial guess.
    grad (callable): Gradient of the objective function.
    hess (callable): Hessian matrix of the objective function.
    eps (float, optional): Tolerance for convergence. Defaults to 1e-7.
    method (str, optional): Method for step length selection. Options are 'standard', 'dichotomy', and 'wolfe'. Defaults to 'standard'.
    callback (callable, optional): Function to call after each iteration. Defaults to None.

    Returns:
    dict: Dictionary containing the optimized solution and other information.
          - 'x': Optimized solution.
          - 'fun': Value of the objective function at the optimized solution.
          - 'it': Number of iterations.
    """
    methods = {
        'dichotomy': dichotomy,
        'standard': lambda *args: 1,
        'wolfe': lambda *args: 1,
    }

    it = 0

    while True:
        it += 1

        if callback:
            callback((x, f(x)))

        gradient = grad(x)
        hessian_inv = np.linalg.inv(hess(x))

        delta = hessian_inv.dot(np.transpose(gradient))

        x_prev = x
        x = x - methods[method](f, x, delta, grad) * delta

        # || delta || < eps
        """
        if np.linalg.norm(delta) < eps:
            return {'x': x, 'fun': f(x), 'it': it}
        """

        # || x_{k} - x_{k - 1} || < eps
        if np.linalg.norm(x - x_prev) < eps:
            return {'x': x, 'fun': f(x), 'it': it}


# Main.Task №2
def dichotomy(f, x, grad, _, eps=1e-7):
    """
    Perform dichotomy method for step length selection in Newton's method.

    Parameters:
    f (callable): Objective function.
    x (ndarray): Current point.
    grad (ndarray): Gradient of the objective function at the current point.
    eps (float, optional): Tolerance for convergence. Defaults to 1e-7.

    Returns:
    float: Optimal step length.
    """
    l = eps
    r = 50.0
    delta = eps / 2

    def foo(xi):
        return [x[i] - xi * grad[i]
                for i in range(len(x))]

    while not r - l < eps:
        x1 = (l + r - delta) / 2
        x2 = (l + r + delta) / 2
        if f(foo(x1)) < f(foo(x2)):
            r = x2
        else:
            l = x1

    return (r + l) / 2


# Main.Task №3 (using scipy.optimize)
def scipy_newton_method(f, x, jac, eps=1e-5,
                        method='Newton-CG', callback=None):
    """
    Perform optimization using Newton's method implemented in scipy.optimize.minimize.

    Parameters:
    f (callable): Objective function.
    x (ndarray): Initial guess.
    jac (callable): Jacobian (gradient) of the objective function.
    eps (float, optional): Tolerance for convergence. Defaults to 1e-5.
    method (str, optional): Method for optimization. Options are 'BFGS' and 'Newton-CG'. Defaults to 'Newton-CG'.
    callback (callable, optional): Function to call after each iteration. Defaults to None.

    Returns:
    OptimizeResult: Result object returned by scipy.optimize.minimize.
    """
    assert method in ('BFGS', 'Newton-CG')
    return minimize(f, x, jac=jac, tol=eps,
                    method=method, callback=callback)


# Additional.Task №1
def wolfe(func, grad, x0, p):
    """
    Perform Wolfe condition check for step length selection.

    Parameters:
    func (callable): Objective function.
    grad (callable): Gradient of the objective function.
    x0 (ndarray): Current point.
    p (ndarray): Search direction.

    Returns:
    float: Optimal step length satisfying the Wolfe conditions.
    """
    alpha, _, _, _ = line_search(func, grad, x0, p)
    if alpha is None:
        alpha = 0
    return alpha


def level_lines(x, surface_func, bounds=(-5, 5), num=100):
    """
    Plot level lines of a surface.

    Parameters:
    x (ndarray): Points to be plotted.
    surface_func (callable): Function defining the surface.
    bounds (tuple, optional): Bounds for plotting. Defaults to (-5, 5).
    num (int, optional): Number of points for plotting. Defaults to 100.
    """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title('Level lines')

    xgrid, ygrid = np.meshgrid(np.linspace(bounds[0], bounds[1], num),
                               np.linspace(bounds[0], bounds[1], num))

    norm = Normalize(vmin=0, vmax=len(x))
    for i in range(len(x)):
        color = plt.cm.cool(norm(i))
        plt.scatter(*x[i], s=40, color=color)

    axes.contour(xgrid, ygrid, surface_func([xgrid, ygrid]))


def create_surface(x_val, fun_val, surface_func, bounds=(-5, 5), num=100):
    """
    Create a 3D surface plot.

    Parameters:
    x_val (list): List of points.
    fun_val (list): Values of the objective function at the points.
    surface_func (callable): Function defining the surface.
    bounds (tuple, optional): Bounds for plotting. Defaults to (-5, 5).
    num (int, optional): Number of points for plotting. Defaults to 100.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], num),
                       np.linspace(bounds[0], bounds[1], num))
    Z = surface_func(np.array([X, Y]))

    ax.plot_surface(X, Y, Z, cmap='BuPu')
    colors = np.linspace(0, 1, len(x_val))

    for i in range(len(x_val)):
        ax.plot(*x_val[i], fun_val[i], '.',
                label='top', zorder=4, markersize=5, c=plt.cm.cool(colors[i]))


def main():
    """
    :MAIN:
    Newton's method:
    Rosenbrock function:
    f = (1 - x)^2 + 100 * (y - x^2)^2
    Polynomial function:
    f = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    f = (x + 2 * y - 7)^2 + (2 * x + y - 5)^2
    Non-polynomial function:
    f = sin(x + y) + (x - y)^2 - 1.5 * x + 2.5 * y + 1
    f = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))

    Newton's method (scipy.optimize.minimize realization):
    Rosenbrock function:
    f = (1 - x)^2 + 100 * (y - x^2)^2
    Polynomial function:
    f = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    f = (x + 2 * y - 7)^2 + (2 * x + y - 5)^2
    Non-polynomial function:
    f = sin(x + y) + (x - y)^2 - 1.5 * x + 2.5 * y + 1
    f = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))
    """

    x, y = Symbol('x'), Symbol('y')
    foo = (1 - x)**2 + 100 * (y - x ** 2)**2
    # foo = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    # foo = (x + 2 * y - 7)**2 + (2 * x + y - 5)**2
    # foo = sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
    # foo = -cos(x) * cos(y) * exp(-((x - pi)**2 + (y - pi)**2))

    f = lambdify((x, y), foo)

    f_x = lambdify((x, y), foo.diff(x))
    f_y = lambdify((x, y), foo.diff(y))

    f_x_x = lambdify((x, y), foo.diff(x).diff(x))
    f_x_y = lambdify((x, y), foo.diff(x).diff(y))
    f_y_x = lambdify((x, y), foo.diff(y).diff(x))
    f_y_y = lambdify((x, y), foo.diff(y).diff(y))

    def fun(vec):
        return f(*vec)

    def fun_jac(vec):
        return np.array([
            f_x(*vec),
            f_y(*vec)
        ])

    def fun_hess(vec):
        return np.array([
            [f_x_x(*vec), f_x_y(*vec)],
            [f_y_x(*vec), f_y_y(*vec)]
        ])

    x_val, fun_val = [], []

    def callback(hist):
        x_val.append(hist[0])
        fun_val.append(hist[1])

    result = newton_method(fun, np.array([-1, -1]), fun_jac,
                           fun_hess, method='wolfe', callback=callback)

    print(result)

    level_lines(x_val, lambda vec: fun(vec))

    create_surface(x_val, fun_val, lambda vec: fun(vec))

    plt.show()


if __name__ == '__main__':
    main()
