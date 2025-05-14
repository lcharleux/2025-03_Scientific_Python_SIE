# A FUNCTION THAT INTEGRATES AND ODE USING RUNGE KUTTA METHOD

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit


def make_runge_kutta(f, jit=True):
    """
    Creates a Runge-Kutta method for solving ordinary differential equations (ODEs)
    for a given function `f`. Optionally, the function and the resulting Runge-Kutta
    solver can be JIT-compiled using Numba for improved performance.

    Parameters:
    -----------
    f : callable
        The function defining the ODE. It must take two arguments:
        - `t` (float): The current time.
        - `y` (array-like): The current state of the system.
        It should return the derivative of `y` with respect to `t`.

    jit : bool, optional (default=True)
        If True, the function `f` and the generated Runge-Kutta solver
        will be JIT-compiled using Numba for faster execution.

    Returns:
    --------
    runge_kutta : callable
        A function that solves the ODE using the 4th-order Runge-Kutta method.
        The returned function has the following signature:
        `runge_kutta(y0, t0, t1, h)` where:
        - `y0` (array-like): Initial state of the system.
        - `t0` (float): Initial time.
        - `t1` (float): Final time.
        - `h` (float): Step size for the integration.
        It returns:
        - `t` (numpy.ndarray): Array of time points.
        - `y` (numpy.ndarray): Array of solution values at each time point.

    Example:
    --------
    >>> def f(t, y):
    >>>     return -2 * y + 1
    >>> rk = make_runge_kutta(f, jit=False)
    >>> y0 = [0]
    >>> t0, t1, h = 0, 5, 0.1
    >>> t, y = rk(y0, t0, t1, h)
    >>> print(t, y)
    """
    if jit:
        f = njit(f)

    def runge_kutta(y0, t0, t1, h):
        n = int((t1 - t0) / h)
        t = np.linspace(t0, t1, n + 1)
        y = np.zeros((n + 1, len(y0)))
        y[0] = y0

        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
            k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
            k4 = f(t[i] + h, y[i] + h * k3)
            y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return t, y

    if jit:
        runge_kutta = njit(runge_kutta)
    return runge_kutta


def f(t, y):
    """
    Example ODE function.
    dy/dt = -2y + 1
    """
    return -2 * y + 1


if __name__ == "__main__":
    # Create Runge-Kutta solvers
    rk = make_runge_kutta(f, jit=False)
    jrk = make_runge_kutta(f, jit=True)

    # Initial conditions
    y0 = [0]
    t0 = 0
    t1 = 5
    h = 0.1
    # Solve the ODE using Runge-Kutta method
    t, y = rk(y0, t0, t1, h)
    t, y = jrk(y0, t0, t1, h)

    # Plot the results
    plt.plot(t, y)
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.title("Runge-Kutta Method")
    plt.grid()
    plt.show()
