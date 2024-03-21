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

    res_x, res_y, res_z = [], [], []

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
        res_x.append(x0)
        res_y.append(y0)
        res_z.append(f(x0, y0))
        if np.abs(f(x_k, y_k) - f(x0, y0)) < eps:
            break
        x0, y0 = x_k, y_k
        
    return (res_x, res_y, res_z, 
            {'x': (round(x_k, 5), round(y_k, 5)), 'fun': round(f(x_k, y_k), 5)})

# Main.Task №2
def golden_ratio(f, l, r, eps=0.0001):
    phi = (1 + np.sqrt(5)) / 2

    while not np.abs(r - l) < eps:
        x1 = r - (r - l) / phi
        x2 = l + (r - l) / phi
        if f(x1) < f(x2):
            r = x2
        else:
            l = x1
            
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


def level_lines(x, y,surface_func, bounds=(-4, 4), num=1000):
    fig = plt.figure()
    axes = fig.add_subplot()
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title('Level lines')
    
    xgrid, ygrid = np.meshgrid(np.linspace(bounds[0], bounds[1], num),
                       np.linspace(bounds[0], bounds[1], num))
    plt.scatter(x, y, s=0.1, color='red')
    axes.contour(xgrid, ygrid, surface_func(xgrid, ygrid))


def create_surface(res_x, res_y, res_z, surface_func, bounds=(-4, 4), num=100):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], num),
                       np.linspace(bounds[0], bounds[1], num))
    Z = surface_func(X, Y)

    ax.plot_surface(X, Y, Z, linewidth=0.1, cmap=plt.cm.coolwarm)
    ax.plot(res_x, res_y, res_z, 'r.', label='top', zorder=4, markersize=5)


def main():
    """
    :MAIN:
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

    :EXTRA:
    Poorly conditioned function:
    f = 100 * x^2 + y^2
    standard :REJECT:
    golden_ratio :SUCCESS:
    dichotomy :SUCCESS:

    Noisy function:
    f = (x^2 + y^2) + random.uniform(0, 0.1)

    Multimodal function:
    f = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    standard :SUCCESS:
    start = (-1, 1) - 14 iterations
    start = (0, -4) - 20 iterations
    golden_ratio :SUCCESS:
    start = (-1, 1) - 215 iterations
    start = (0, -4) - 382 iterations
    dichotomy :SUCCESS:
    start = (-1, 1) - 215 iterations
    start = (0, -4) - 382 iterations
    """
    
    res_x, res_y, res_z, answer = gradient_descent(
        '(x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2', -1, 1, method='golden_ratio')

    create_surface(res_x, res_y, res_z,
                   lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2, bounds=(-4, 4))
    
    level_lines(res_x, res_y, lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2)
    
    print(answer)
    
    plt.show()


if __name__ == '__main__':
    main()
