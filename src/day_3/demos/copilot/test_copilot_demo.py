import numpy as np
import pytest
from copilot_demo import make_runge_kutta


def test_make_runge_kutta():
    # Define a simple ODE: dy/dt = -2y + 1
    def f(t, y):
        return -2 * y + 1

    # Create Runge-Kutta solvers
    rk = make_runge_kutta(f, jit=False)
    jrk = make_runge_kutta(f, jit=True)

    # Initial conditions
    y0 = np.zeros(1)  # Initial condition as a 2D array
    t0 = 0
    t1 = 1
    h = 0.001

    # Solve the ODE
    t_rk, y_rk = rk(y0, t0, t1, h)
    t_jrk, y_jrk = jrk(y0, t0, t1, h)

    # Expected solution (analytical solution of dy/dt = -2y + 1)
    expected_t = np.linspace(t0, t1, int((t1 - t0) / h) + 1)
    expected_y = 0.5 * (1 - np.exp(-2 * expected_t))

    # Assert that the numerical solutions are close to the analytical solution
    np.testing.assert_allclose(y_rk.flatten(), expected_y, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(y_jrk.flatten(), expected_y, rtol=1e-5, atol=1e-8)
