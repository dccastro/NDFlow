from typing import Callable, List, TypeVar

import numpy as np

T = TypeVar('T')
DT = TypeVar('DT')


def _step_euler(f: Callable[[T, float], DT], y: T, t: float, dt: float,
                update: Callable[[T, DT, float], T]) -> T:
    return update(y, f(y, t), dt)


def _step_midpoint(f: Callable[[T, float], DT], y: T, t: float, dt: float,
                   update: Callable[[T, DT, float], T]) -> T:
    y_ = update(y, f(y, t), dt / 2)
    return update(y, f(y_, t + dt / 2), dt)


def _step_rk4(f: Callable[[T, float], DT], y: T, t: float, dt: float,
              update: Callable[[T, DT, float], T]) -> T:
    k1 = f(y, t)
    k2 = f(update(y, k1, dt / 2), t + dt / 2)
    k3 = f(update(y, k2, dt / 2), t + dt / 2)
    k4 = f(update(y, k3, dt), t + dt)
    y_ = update(y, k1, dt / 6)
    y_ = update(y_, k2, dt / 3)
    y_ = update(y_, k3, dt / 3)
    return update(y_, k4, dt / 6)


def integrate(f: Callable[[T, float], DT], y0: T,
              update: Callable[[T, DT, float], T],
              t0=0., t_max=1., dt=.001, t=None, method='rk4'):
    """
    Numerically solve ordinary differential equations (ODEs) of the form :math:`y'(t) = f(y, t)`, with
    initial conditions :math:`y(0) = y_0`.

    Parameters
    ----------
    f : (T, float) -> DT
        Derivative: :math:`f(y, t) = y'(t)`
    y0 : T
        Initial conditions: :math:`y_0 = y(0)`
    update : (T, DT, float) -> T
        Function which applies the update, e.g. `update(y, f, dt) = y + f * dt`
    t0 : float
        Initial timestamp
    t_max : float
        Upper limit of integration
    dt : float
        Step size
    t : array-like
        Sequence of timestamps. If given, `t0`, `t_max` and `dt` are ignored.
    method : {'euler', 'midpoint', 'rk4'}
        * 'euler': Euler method (1st order)
        * 'midpoint': Midpoint method (2nd order)
        * 'rk4' (default): classical Runge-Kutta method (4th order)

    Returns
    -------
    (List[T], np.ndarray)
        List of simulated states and timestamps
    """
    if method == 'euler':
        step = _step_euler
    elif method == 'midpoint':
        step = _step_midpoint
    elif method == 'rk4':
        step = _step_rk4
    else:
        raise NotImplementedError("Method '%s' not implemented" % method)

    if t is None:
        t = np.arange(t0, t_max + dt, dt)
    N = len(t)
    y = [None] * N  # type: List[T]
    y[0] = y0
    for i in range(N - 1):
        y[i + 1] = step(f, y[i], t[i], t[i + 1] - t[i], update)
    return y, t
