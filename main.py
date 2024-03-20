import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sympy import Symbol, lambdify, parsing


# Main.Task №1 and №2
def gradient_descent(foo, x0, y0, step=0.1, eps=0.0001, method='standard'):
    x, y = Symbol('x'), Symbol('y')
    parsed = parsing.parse_expr(foo)
    f = lambdify((x, y), parsed)
    f_x = lambdify((x, y), parsed.diff(x))
    f_y = lambdify((x, y), parsed.diff(y))

    methods = {
        'standard': lambda x0, y0: (x0 - step * f_x(x0, y0), y0 - step * f_y(x0, y0)),
        'golden_ratio': lambda x0, y0: (
            golden_ratio(lambda x_val: f(x_val, y0), x0 - step, x0 + step, eps),
            golden_ratio(lambda y_val: f(x0, y_val), y0 - step, y0 + step, eps)
        ),
        'dichotomy': lambda x0, y0: (
            dichotomy(lambda x_val: f(x_val, y0), x0 - step, x0 + step, eps),
            dichotomy(lambda y_val: f(x0, y_val), y0 - step, y0 + step, eps)
        )
    }

    while True:
        x_k, y_k = methods[method](x0, y0)
        if np.abs(f(x_k, y_k) - f(x0, y0)) < eps:
            return {'x': (round(x_k, 5), round(y_k, 5)), 'fun': round(f(x_k, y_k), 5)}
        x0, y0 = x_k, y_k


# Main.Task №2
def golden_ratio(f, l, r, eps=0.0001):
    phi = (1 + np.sqrt(5)) / 2

    while not np.abs(r - l) < eps:
        x1 = r - (r - l) / phi
        x2 = l + (r - l) / phi
        if f(x1) >= f(x2):
            l = x1
        else:
            r = x2

    # return {'x': round((r + l) / 2, 5), 'fun': round(f((r + l) / 2), 5)}
    return round((r + l) / 2, 5)


# Main.Task №3
def nelder_mead(foo, x0, y0):
    return minimize(foo, np.array([x0, y0]), method='Nelder-Mead')


# Additional.Task №1
def dichotomy(f, l, r, eps=0.0001):
    delta = eps / 2
    while not np.abs(r - l) < eps:
        x1 = (l + r - delta) / 2
        x2 = (l + r + delta) / 2
        if f(x1) < f(x2):
            r = x2
        else:
            l = x1
    return round((r + l) / 2, 5)


def create_plot(plot_func, bounds=(-10, 10), num=100):
    fig, ax = plt.subplots()

    X = np.linspace(bounds[0], bounds[1], num)
    Y = plot_func(X)

    ax.plot(X, Y)


def create_surface(surface_func, bounds=(-1, 1), num=100):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], num),
                       np.linspace(bounds[0], bounds[1], num))
    Z = surface_func(X, Y)

    ax.plot_surface(X, Y, Z, linewidth=0.1, cmap=plt.cm.coolwarm)


def main():
    """
    Gradient descent:
    f = x^2 + y^2 :SUCCESS:
    f = x^2 * y^2 * ln(4 * x^2 + y^2) :SUCCESS:
    f = -0.83 * x^2 - 0.23 * y^2 :REJECT: (-infinity)

    One dimensional search (Golden-section search):
    f = (x - 2)^2 + 4 :SUCCESS:

    Nelder-Mead (scipy.optimize.minimize realization):
    f = x^2 + y^2 :SUCCESS:
    f = x^2 * y^2 * ln(4 * x^2 + y^2) :SUCCESS:
    f = -0.83 * x^2 - 0.23 * y^2 :REJECT: (-infinity)
    """

    create_plot(lambda x: (x - 2) ** 2 + 4)
    print(golden_ratio(lambda x: (x - 2) ** 2 + 4, -10, 10))

    create_surface(lambda x, y: x ** 2 + y ** 2)
    print(gradient_descent('x**2 * y**2 * ln(4 * x**2 + y**2)', -1, 1))
    print(gradient_descent('x**2 * y**2 * ln(4 * x**2 + y**2)', -1, 1, method='golden_ratio'))
    print(gradient_descent('x**2 * y**2 * ln(4 * x**2 + y**2)', -1, 1, method='dichotomy'))
    print(nelder_mead(lambda x: x[0] ** 2 + x[1] ** 2, -1, 1))

    plt.show()


if __name__ == '__main__':
    main()
