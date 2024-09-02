import numpy as np
import matplotlib.pyplot as plt

'''
delta(u)/delta(t) + c * delta(u)/delta(x) = f(x,t)
c = 2, f(x,t) = cos(xt)
u(x,0) = sin(x) = phi(x)
u(0,t) = 1 - cos(t) = psi(t)
0 <= x <= a
0 <= t <= T
'''

c = 2
phi = np.sin
psi = lambda x: 1 - np.cos(x)
f = lambda x, t: np.cos(x * t)


# Явная схема
def explicit_scheme(c, f, xs, ts, u_mesh, tau, h):
    kappa = c * tau / h
    u_mesh = u_mesh.copy()
    n = xs.shape[0]
    k = ts.shape[0]
    for i in range(1, n):
        for j in range(k - 1):
            # устойчива условно при kappa <= 1
            u_mesh[i, j + 1] = tau * f(xs[i], ts[j]) + (1 - kappa) * u_mesh[i, j] + kappa * u_mesh[i - 1, j]
    return u_mesh


# Чисто неявная схема
def purely_implicit_scheme(c, f, xs, ts, u_mesh, tau, h):
    kappa = c * tau / h
    denum = 1 + kappa
    u_mesh = u_mesh.copy()
    n = xs.shape[0]
    k = ts.shape[0]
    for i in range(1, n):
        for j in range(k - 1):
            # устойчива безусловно
            u_mesh[i, j + 1] = (tau * f(xs[i], ts[j + 1]) + kappa * u_mesh[i - 1, j + 1] + u_mesh[i, j]) / denum
    return u_mesh


# Неявная схема
def implicit_scheme(c, f, xs, ts, u_mesh, tau, h):
    kappa = c * tau / h
    u_mesh = u_mesh.copy()
    n = xs.shape[0]
    k = ts.shape[0]
    for i in range(1, n):
        for j in range(k - 1):
            # устойчива условно при kappa >= 1
            u_mesh[i, j + 1] = tau / kappa * f(xs[i], ts[j + 1]) + 1 / kappa * u_mesh[i - 1, j] + \
                                (1 - 1 / kappa) * u_mesh[i - 1, j + 1]
    return u_mesh


# Симметричная схема
def symmetric_scheme(c, f, xs, ts, u_mesh, tau, h):
    kappa = c * tau / h
    denum = 1 + kappa
    u_mesh = u_mesh.copy()
    n = xs.shape[0]
    k = ts.shape[0]
    for i in range(1, n):
        for j in range(k - 1):
            # устойчива безусловно
            u_mesh[i, j + 1] = 2 * tau / denum * f(xs[i] + h / 2, ts[j] + tau / 2) - (1 - kappa) / \
                                 denum * u_mesh[i - 1, j + 1] + (1 - kappa) / denum * u_mesh[i, j] + u_mesh[i - 1, j]
    return u_mesh


if __name__ == '__main__':
    # Выбираем нужные каппа
    kappas = (0.93, 0.98, 1, 1.02, 1.07)

    # Загоняем функции в кортеж
    funcs = (explicit_scheme, purely_implicit_scheme, implicit_scheme, symmetric_scheme)
    func_names = ("Явная схема", "Чисто неявная схема", "Неявная схема", "Симметричная схема")
    inds = range(4)

    # Создаём нужное количнство графиков
    fig, axes = plt.subplots(len(funcs), len(kappas), figsize=(20, 20), subplot_kw={"projection": "3d"})

    # Выбираем границу по x
    a = 4

    # Выбираем границу по времени
    to_time = 15

    # Выбираем тау
    tau = 0.01

    # Проходим по всем нужным методам и значениям каппа
    for i, func_name, func in zip(inds, func_names, funcs):
        for ax, kappa in zip(axes[i, :], kappas):
            # Рассчитываем нужное h
            h = c * tau / kappa
            # Сетка по x и t
            xs = np.arange(0, a, h)
            ts = np.arange(0, to_time, tau)

            n = xs.shape[0]
            k = ts.shape[0]
            # Сетка значений u
            u_mesh = np.zeros((n, k))
            # Заполняем значения u при t = 0
            u_mesh[:, 0] = np.array(list(map(phi, xs)))
            # Заполняем значения u при x = 0
            u_mesh[0, :] = np.array(list(map(psi, ts)))
            u_values = func(c, f, xs, ts, u_mesh, tau, h)

            ax.set_xlabel('$x$', fontsize=10)
            ax.set_ylabel('$t$', fontsize=10)
            ax.view_init(elev=15, azim=-135)
            xv, tv = np.meshgrid(xs, ts)
            ax.set_title(rf"{func_name}, $\kappa={kappa}$", fontsize=10)
            ax.plot_surface(xv, tv, u_values.T, cmap='viridis',
                            linewidth=0, antialiased=False)
