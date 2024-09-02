# Вариант 15

import numpy as np
import pandas as pd
import sympy

from sympy.abc import x


# массив многочленов Якоби
def jacoby_polynoms(k, n):
    p = [x] * (n + 1)

    for j in range(n + 1):
        if j == 0:
            p[j] = 1
        elif j == 1:
            p[j] = (1 + k) * x
        else:
            tmp_3 = (j + 2 * k) * j
            tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
            tmp_2 = (j + k) * (j + k - 1)
            p[j] = (tmp_1 * x * p[j - 1] - tmp_2 * p[j - 2]) / tmp_3
    return p


# значения многочленов Якоби
def jacoby_polynoms_val(k, n, x_0):
    p = [0] * (n + 1)

    for j in range(n + 1):
        if j == 0:
            p[j] = 1
        elif j == 1:
            p[j] = (1 + k) * x_0
        else:
            tmp_3 = (j + 2 * k) * j
            tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
            tmp_2 = (j + k) * (j + k - 1)
            p[j] = (tmp_1 * x_0 * p[j - 1] - tmp_2 * p[j - 2]) / tmp_3
    return p


# координатные функции и их производные
def coord_functions(k, n):
    omega = [x] * n
    der_omega = [x] * n
    second_der_omega = [x] * n

    jacs = jacoby_polynoms(k, n)
    djacs = jacoby_polynoms(k - 1, n + 1)

    for i in range(n):
        omega[i] = (1 - x ** 2) * jacs[i]
        omega[i] = sympy.simplify(omega[i])

        der_omega[i] = (-2) * (i + 1) * (1 - x ** 2) ** (k - 1) * djacs[i + 1]
        der_omega[i] = sympy.simplify(der_omega[i])

        tmp1 = (-2) * (k - 1) * (1 - x ** 2) ** (k - 2) * djacs[i + 1]
        tmp2 = (1 - x ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jacs[i]
        second_der_omega[i] = (-2) * (i + 1) * (tmp1 + tmp2)
        second_der_omega[i] = sympy.simplify(second_der_omega[i])

    return omega, der_omega, second_der_omega


def coord_functions_val(k, n, x_0):
    omega = [0] * n
    der_omega = [0] * n
    second_der_omega = [0] * n

    jacs = jacoby_polynoms_val(k, n, x_0)
    djacs = jacoby_polynoms_val(k - 1, n + 1, x_0)

    for i in range(n):
        omega[i] = (1 - x_0 ** 2) * jacs[i]
        omega[i] = sympy.simplify(omega[i])

        der_omega[i] = (-2) * (i + 1) * (1 - x_0 ** 2) ** (k - 1) * djacs[i + 1]
        der_omega[i] = sympy.simplify(der_omega[i])

        tmp1 = (-2) * (k - 1) * (1 - x_0 ** 2) ** (k - 2) * djacs[i + 1]
        tmp2 = (1 - x_0 ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jacs[i]
        second_der_omega[i] = (-2) * (i + 1) * (tmp1 + tmp2)
        second_der_omega[i] = sympy.simplify(second_der_omega[i])

    return omega, der_omega, second_der_omega


def ritz_method(k, n):
    omegas, domegas, ddomegas = coord_functions(k, n)
    x = sympy.symbols('x')
    f = 2 - x

    # Создаем матрицу и вектор для построения системы линейных уравнений метода Ритца
    A = np.zeros((n, n))
    b = np.zeros((n, 1))

    # 1) Вычисляем произведение f(x) на одну из координатных функций
    # 2) Вычисляем значение интеграла от h на [-1,1] (элементы вектора правой части)
    for i in range(3):
        h = f * omegas[i]
        b[i] = sympy.integrals.integrate(h, (x, -1, 1))

    # Задаем значения узлов и соответствующие веса для формулы Гаусса
    x1 = 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7))
    x2 = 1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))
    c1 = (322 + 13 * np.sqrt(70)) / 900
    c2 = (322 - 13 * np.sqrt(70)) / 900
    x_i = [-x2, -x1, 0, x1, x2]
    c_i = [c2, c1, 128 / 225, c1, c2]

    # Вычисляем значения координатных функция и их производных в этом узле
    values = []
    for i in range(5):
        omegas_val, domegas_val, ddomegas_val = coord_functions_val(k, n, x_i[i])
        values.append([omegas_val, domegas_val])

    # Вычисление интеграла по формуле Гаусса
    def gauss_method(nodes, coefs, i, j):
        s = 0
        # Перебираем все узлы формулы Гаусса
        for k in range(len(nodes)):
            tmp_1 = (((nodes[k] + 4) / (nodes[k] + 5)) * values[k][1][j] * values[k][1][i] + np.exp(nodes[k] / 4) *
                     values[k][0][i] * values[k][0][j])
            s += coefs[k] * tmp_1
        return s

    for i in range(n):
        for j in range(n):
            A[i][j] = gauss_method(x_i, c_i, i, j)
    # Вычисляем коэффициенты для решения методом Ритца
    coeffs = np.linalg.solve(A, b)
    return coeffs, A, b


def cheb_nodes(n, a, b):
    result = []
    for i in range(1, n + 1):
        tmp1 = (1 / 2) * (a + b)
        tmp2 = (1 / 2) * (b - a)
        result.append(np.cos((2 * i - 1) * np.pi / (2 * n)))
    return result


# Метод коллокации
def collocation_method(k, n):
    # Генерируем узлы метода коллокации с помощью Чебышева
    nodes = cheb_nodes(n, -1, 1)

    # Функции, которые представляют правую часть дифура и коэффициенты при производных в уравнении
    f = lambda x: 2 - x
    p = lambda x: (x + 4) / (x + 5)
    dp = lambda x: 1 / (x + 5) ** 2
    r = lambda x: np.exp(x / 4)

    # Матрица и вектор для построения системы линейных уравнений метода коллокации
    A = np.zeros((n, n))
    b = np.zeros((n, 1))

    for i in range(n):
        # Значение правой части в узле
        b[i] = f(nodes[i])
        # Значения координатных функций и их производных в узле
        omega, domega, ddomega = coord_functions_val(k, n, nodes[i])
        for j in range(n):
            tmp1 = p(nodes[i]) * ddomega[j]
            tmp2 = dp(nodes[i]) * domega[j]
            tmp3 = r(nodes[i]) * omega[j]
            A[i][j] = (-1) * (tmp1 + tmp2) + tmp3
    coeffs = np.linalg.solve(A, b)
    return coeffs, A, b


def solution(coeffs, dots):
    x1, x2, x3 = dots[0], dots[1], dots[2]

    exact_value = [0.721373, 0.813764, 0.541390]
    result = [0] * 3
    n = len(coeffs)

    omega_x1 = coord_functions_val(1, n, x1)[0]
    omega_x2 = coord_functions_val(1, n, x2)[0]
    omega_x3 = coord_functions_val(1, n, x3)[0]

    for i in range(3):
        result[0] += coeffs[i] * omega_x1[i]
        result[1] += coeffs[i] * omega_x2[i]
        result[2] += coeffs[i] * omega_x3[i]

    errors = [exact_value[k] - result[k] for k in range(3)]
    final_result = []
    for j in range(3):
        final_result.append(round(result[j][0], 5))
    for k in range(3):
        final_result.append(round(errors[k][0], 5))
    return final_result


def create_table(values):
    column = [
        "y(-0.5)",
        "y(0)",
        "y(0.5)",
        "(y*)-y(-0.5)",
        "(y*)-y(0)",
        "(y*)-y(0.5)"
    ]
    indexes = [3, 4, 5, 6, 7]
    table = pd.DataFrame(data=values, columns=column, index=indexes)
    table.columns.name = "n"
    return table


if __name__ == '__main__':
    dots = [-0.5, 0.0, 0.5]
    val_Ritz = []
    coeffs, A, b = [], [], []
    print("Метод Ритца")
    for i in range(3, 8):
        coeffs, A, b = ritz_method(1, i)
        val_Ritz.append(solution(coeffs, dots))
    print("Расширенная матрица системы (n = 7):")
    print("А = ", A)
    print("Число обусловленности матрицы А = ", np.linalg.cond(A))
    print("b = ", b)
    print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coeffs)
    result_table = create_table(val_Ritz)
    print(result_table)

    val_colloc = []
    for i in range(3, 8):
        coeffs, A, b = collocation_method(1, i)
        val_colloc.append(solution(coeffs, dots))
    result_table = create_table(val_colloc)
    print("\n\nМетод коллокации")
    print("Расширенная матрица системы:")
    print("А = ", A)
    print("Число обусловленности матрицы А = ", np.linalg.cond(A))
    print("b = ", b)
    print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coeffs)
    print(result_table)
