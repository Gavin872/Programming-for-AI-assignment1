import numpy as np
from numcompute.optim import grad, jacobian


def test_grad_central():
    def f(x):
        return x[0] ** 2 + 3 * x[1] ** 2

    x = np.array([2.0, 3.0])
    g = grad(f, x, method="central")
    assert np.allclose(g, np.array([4.0, 18.0]), atol=1e-4)


def test_grad_forward():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([3.0, 4.0])
    g = grad(f, x, method="forward")
    assert np.allclose(g, np.array([6.0, 8.0]), atol=1e-3)


def test_jacobian():
    def F(x):
        return np.array([x[0] + x[1], x[0] * x[1]])

    x = np.array([2.0, 3.0])
    J = jacobian(F, x)
    expected = np.array([[1.0, 1.0], [3.0, 2.0]])
    assert np.allclose(J, expected, atol=1e-4)