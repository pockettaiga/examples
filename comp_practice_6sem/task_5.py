# Вариант 5

import numpy as np
import pandas as pd
from scipy import integrate


def phi(x):
    return np.sin(np.pi * x * (x - 1) / 3)


def psi(x, p):
    return np.sqrt(2) * np.sin(np.pi * x * p)


# Аналитическое решение, используя ряд Фурье
# Вычисляем коэффициенты разложения по методу Фурье
def coefs_fou(basis_funcs):
    coefs = []
    for p in range(1, basis_funcs):
        coefs.append(integrate.quad(lambda x: phi(x) * psi(x, p), 0, 1)[0])
    return coefs


# Вычисляем решение уравнения в точке (x,t) с использованием коэфов метода Фурье
# (вычисляем сумму всех базисных функций, взвешенных соответствующими коэффициентами и умноженных на экспоненту)
def fou_solve(coefs, x, t):
    res = 0
    for p in range(len(coefs)):
        res += coefs[p] * np.exp(-np.pi ** 2 * (p + 1) ** 2 * t) * psi(x, p + 1)
    return res


# Аналитическое решение, используя дискретный ряд Фурье
# Вычисляем коэффициенты разложения методом конечных разностей
# (используем сумму значений базисных функций phi(x), psi(x),
# взвешенных по координате x и масштабированных на шаг сетки h)
def coefs_discrete_Fou(N):
    h = 1 / N
    coefs = []
    for p in range(1, N):
        coef = 0
        for i in range(1, N):
            coef += phi(i * h) * psi(i * h, p)
        coef *= h
        coefs.append(coef)
    return coefs


# Вычисляем решение уравнения с использованием коэффициентов метода конечных разностей
# (вычисляем сумму всех базисных функций, взвешенных соответствующими коэффициентами и масштабированных экспонентой)
def discrete_fou_solve(coefs, x, t):
    res = 0
    for p in range(len(coefs)):
        res += coefs[p] * np.exp(-np.pi ** 2 * (p + 1) ** 2 * t) * psi(x, p + 1)
    return res


# Сеточное решение, используя ДРФ по схеме с весами при N = 5, 10, 20
# Вычисляем параметры lambda_p для каждого p для схемы Кранка-Никольсона
# Принимаем количество узлов N и количество временных шагов M и параметра сигма
def lamb(N, M, sigma):
    h = 1 / N
    tau = 0.1 / M
    l = []
    for p in range(1, N):
        tmp1 = 1 - (4 * (1 - sigma) * tau / h ** 2) * np.sin(p * np.pi * h / 2) ** 2
        tmp2 = 1 + (4 * sigma * tau / h ** 2) * np.sin(p * np.pi * h / 2) ** 2
        l.append(tmp1 / tmp2)
    return l


# Вычисляем значение решения с помощью суммы по всем базисным функциям с учетом параметров схемы
def grid_fou_solve(N, M, coefs, l, x, t):
    h = 1 / N
    tau = 0.1 / M
    res = 0
    k = t / tau
    for p in range(len(coefs)):
        res += coefs[p] * (l[p] ** k) * psi(x, p + 1)
    return res


# Вычисляем максимальную ошибку численного решения для заданных параметров сетки и коэффициентов сигма
# Затем сравнением решение с аналитическим решением или решением, полученным методом Фурье
def max_error(N, M, sigma, x_arr, t_arr):
    u_grid = []
    error = []
    for i in range(len(t_arr)):
        arr_u_grid = []
        arr_error = []
        for j in range(len(x_arr)):
            temp = grid_fou_solve(N, M, coefs_discrete_Fou(N), lamb(N, M, sigma), x_arr[j], t_arr[i])
            arr_u_grid.append(temp)
            if t_arr[i] == 0:
                arr_error.append(abs(temp - phi(x_arr[j])))
            else:
                arr_error.append(abs(temp - fou_solve(coefs_fou(10), x_arr[j], t_arr[i])))
        u_grid.append(arr_u_grid)
        error.append(arr_error)

    return max(max(error))


def sigma(h_tau_index):
    return np.array([0, 1, 0.5, 0.5 - 1 * h_tau[h_tau_index][0] ** 2 / (12 * h_tau[h_tau_index][1])])


if __name__ == '__main__':
    T = 2

    # Кол-во базисных функций, используемых для разложения
    basis_funcs = 15

    x_arr = np.linspace(0, 1, 6)
    t_arr = np.linspace(0, T, 6)

    # Вычисляем решение уравнения для каждой комбинации (x,t), если t=0, то используем начальное условие phi(x),
    # иначе используем метод Фурье
    uf = []
    for t in t_arr:
        arr = []
        for x in x_arr:
            if t == 0:
                arr.append(phi(x))
            else:
                arr.append(fou_solve(coefs_fou(basis_funcs), x, t))
        uf.append(arr)

    uf_table = pd.DataFrame(data=uf, index=t_arr, columns=x_arr)
    uf_table.columns.name = "t \\ x"
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(uf_table)

    # Кол-во узлов в сетке для метода конечных разностей
    N = 10

    # Вычисляем решение уравнения с использованием метода конечных разностей для каждой комбинации (x,t) и
    # ошибку численного решения, сравнивая с результами, полученными методом Фурье
    udsf = []
    error = []
    for i in range(len(t_arr)):
        arr_udsf = []
        arr_error = []
        for j in range(len(x_arr)):
            temp = discrete_fou_solve(coefs_discrete_Fou(N), x_arr[j], t_arr[i])
            arr_udsf.append(temp)
            arr_error.append(abs(temp - fou_solve(coefs_fou(15), x_arr[j], t_arr[i])))
        udsf.append(arr_udsf)
        error.append(arr_error)

    # Выводим число узлов и максимальную ошибку численного решения
    print("N =", N)
    print("||uf - udsf^(N)|| = max|uf - udsf^(N)| = ", max(max(error)))

    N_M = [[5, 5], [10, 20], [20, 80], [20, 20]]
    h_tau = [[0.2, 0.02], [0.1, 0.005], [0.05, 0.00125], [0.05, 0.005]]

    x_arr = np.linspace(0, 1, 6)
    t_arr = np.linspace(0, T, 6)

    result = []
    for i in range(len(N_M)):
        sigmas = sigma(i)
        arr_result = []
        for j in range(len(sigmas)):
            arr_result.append(max_error(N_M[i][0], N_M[i][1], sigmas[j], x_arr, t_arr))
        result.append(arr_result)

    result = np.array(result).transpose()

    # Сводка максимальной ошибки для различных комбинаций значений параметров сетки
    error_grid_table = pd.DataFrame(data=result,
                                    index=['ДРФ σ = 0', 'ДРФ σ = 1', 'ДРФ σ = 1/2', 'ДРФ σ = 1/2 - h ** 2 / (12 * τ)'],
                                    columns=['(0.2, 0.02)', '(0.1, 0.005)', '(0.05, 0.00125)', '(0.05, 0.005)'])
    error_grid_table.columns.name = "(h, τ)"
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(error_grid_table)
