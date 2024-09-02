import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numpy.linalg import inv

import numpy as np

import pandas as pd

# Коэффициент перед интегралом
lambd = -0.5
a = 0
b = 1

f = lambda x: x + 0.5


'''Метод замены ядра на вырожденное'''


# Метод средних прямоугольников
def middle_rects(f, a, b, h):
    s = 0
    i = 0
    while a + i * h <= b:
        s += f(a + (i + 1 / 2) * h)
        i += 1
    return s * h


# Метод вырожденного ядра, n - степень разложения в ряд Маклорена, h - шаг в квадратурной формуле
def singular_kernel(lambd, left, right, n, h):
    alpha = []
    beta = []
    factorial = 1

    # Разложение функции ядра
    for i in range(n):
        if i > 1:
            factorial *= i
        alpha.append(lambda x, i=i: x ** i)
        beta.append(lambda y, i=i, factorial=factorial: lambd * y ** i / factorial)

    # Создаём матрицы A, b
    gamma_matrix = np.zeros((n, n))
    b_vector = np.zeros(n)

    # Заполняем матрицы A, b
    for i in range(n):
        for j in range(n):
            gamma_matrix[i][j] = middle_rects(lambda y: beta[i](y) * alpha[j](y), left, right, h=h)
            b_vector[i] = middle_rects(lambda y: beta[i](y) * f(y), left, right, h=h)
    a_matrix = np.eye(n) - gamma_matrix

    # Находим c из Ac=b
    c = inv(a_matrix).dot(b_vector)

    # Синтезируем u
    u = lambda x: f(x) + sum(c_i * alpha_i(x) for alpha_i, c_i in zip(alpha, c))
    return u


# Вычисление левой части уравнения
def left_side(u, x):
    integral, err_eval = integrate.quad(lambda y: np.exp(x * y) * u(y), a, b)
    return u(x) - lambd * integral


# Невязка
def loss(u, a, b, times=3):
    x = np.linspace(a, b, times)
    left_side_val = np.array(list(map(lambda x: left_side(u, x), x)))
    f_val = np.array(list(map(f, x)))
    return np.max(np.abs(left_side_val - f_val))


'''Метод механических квадратур'''


# Ядро
def kernel(x, y):
    return np.exp(x * y)


# n - число вершин
def mech_quad(lambd, a, b, n):
    # Разбиваем отрезок [a, b] на n + 1 частей
    x = np.linspace(a, b, n + 1)
    h = (b - a) / (n - 1)
    x = x[:-1]
    x_shifted = x + h / 2

    d_matrix = np.eye(n)

    for i in range(n):
        for j in range(n):
            # Вычисляем элементы матрицы D
            d_matrix[i, j] -= lambd * h * kernel(x[i], x_shifted[j])
    # Находим вектор g
    f_vector = np.array(list(map(f, x)))

    # Находим искомый вектор z из уравнения Dz=g
    z_vector = inv(d_matrix).dot(f_vector)

    u = lambda x: lambd * sum(h * kernel(x, x_shifted[k]) * z_vector[k] for k in range(n)) + f(x)
    return u


if __name__ == '__main__':
    # Метод замены ядра на вырожденное
    # Шаги квадратурной формулы
    hs = (1e-1,
          1e-2,
          1e-3,
          1e-4,
          1e-5
          )
    data = {'h': [], 'n': [], 'Невязка': []}
    for h in hs:
        for n in (8, 9, 10):
            u = singular_kernel(lambd, a, b, n=n, h=h)
            data['h'].append(h)
            data['n'].append(n)
            data['Невязка'].append(loss(u, a, b))

    # Запоминаем найденное u
    degen_u = u

    df = pd.DataFrame(data=data)
    print(df)

    # Метод механических квадратур
    # Число вершин
    ns = [10, 20, 30, 50, 100, 200]

    data = {'n': ns, 'Невязка': []}

    for n in ns:
        u = mech_quad(lambd, 0, 1, n)
        current_loss = loss(u, 0, 1)
        data['Невязка'].append(current_loss)

    df = pd.DataFrame(data=data)
    print(df)

    x = np.linspace(a, b, 1000)

    fig, ax = plt.subplots()
    ax.plot(x, u(x), label="Метод механических квадратур")
    ax.plot(x, degen_u(x), label="Метод замены ядра на вырожденное")
    ax.legend()
    plt.show()
