import numpy as np
import scipy.integrate as integrate
import pandas as pd
from scipy.special import jacobi
from numpy.linalg import eig

import matplotlib.pyplot as plt

N = 7

k = 15.7503
l = 19.58201

x0 = -1
x1 = -0.7978525

p = lambda x: k * x + l
q = lambda x: k**2 / (k * x + l) - k**3 * x


# Скалярное произведение f1(x) и f2(x) на [x_0, x_1]
def dotL2(f1, f2):
    return integrate.quad(lambda x: f1(x) * f2(x), x0, x1)[0]


# Интеграл с весом для пары функций w1(x) и w2(x) с их производными dw1(x), dw2(x) и весами p(x), q(x)
def braces(w1, w2, dw1, dw2):
    return integrate.quad(lambda x: p(x) * dw1(x) * dw2(x) + q(x) * w1(x) * w2(x), x0, x1)[0]


# Собственное значение при заданном k
def eigen_val(k, p, q):
    return np.pi**2 * k**2 / (x1 - x0)**2 * p + q


# Собственный вектор для заданного k
def eigen_vec(k):
    ck = np.tan(np.pi / (x1 - x0) * k)
    fun = lambda x: ck * np.cos(np.pi / (x1 - x0) * k * x) + np.sin(np.pi / (x1 - x0) * k * x)
    # Возвращаем собственный вектор, нормированный на единичную длину
    return lambda x: fun(x) / np.sqrt(dotL2(fun, fun))


# Производная собственного вектора для заданного k
def deigen_vec(k):
    ck = np.tan(np.pi / (x1 - x0) * k)
    fun = lambda x: ck * np.cos(np.pi / (x1 - x0) * k * x) + np.sin(np.pi / (x1 - x0) * k * x)
    c = np.sqrt(dotL2(fun, fun))
    a = np.pi / (x1 - x0) * k
    # Возвращаем производную собственной функции и нормализуем на значение c
    return lambda x: (a * np.cos(a * x) - ck * a * np.sin(a * x)) / c


# Вторая производня собственного вектора для заданного k
def ddeigen_vec(k):
    ck = np.tan(np.pi / (x1 - x0) * k)
    fun = lambda x: ck * np.cos(np.pi / (x1 - x0) * k * x) + np.sin(np.pi / (x1 - x0) * k * x)
    c = np.sqrt(dotL2(fun, fun))
    a = np.pi / (x1 - x0) * k
    return lambda x: (-a * a * np.sin(a * x) - ck * a * a * np.cos(a * x)) / c


# Невязка между левой и правой частями уравнения для собственной функции
def loss(i, pm, qm, x):
    # Левая часть уравнения для СФ
    left_side = lambda x: -k * deigen_vec(i)(x) - p(x) * ddeigen_vec(i)(x) + q(x) * eigen_vec(i)(x)
    # Правая часть уравнения для СФ
    right_side = lambda x: eigen_val(i, pm, qm) * eigen_vec(i)(x)
    return np.max(np.abs(right_side(x) - left_side(x)))


# Вычисляем базисные функции и нормируем их
def basic_func(k):
    fun = lambda x: (1 - ((2 * x - x0 - x1) / (x1 - x0))**2) * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))
    c = np.sqrt(dotL2(fun, fun))
    return lambda x: fun(x) / c


# Вычисляем производные базисных функций и нормируем их
# Если k=0, используем формулу для вычисления производной базисной функции 1-го порядка
# Иначе 2-го порядка
def dbasic_func(k):
    fun = lambda x: (1 - ((2 * x - x0 - x1) / (x1 - x0))**2) * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))
    c = np.sqrt(dotL2(fun, fun))
    if k == 0:
        return lambda x: (-4 * (2 * x - x0 - x1) / (x1 - x0)**2 * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))) / c
    return lambda x: (-4 * (2 * x - x0 - x1) / (x1 - x0)**2 * jacobi(k, 2, 2)((2 * x - x0 - x1) / (x1 - x0))
                      + 2 / (x1 - x0) * (1 - ((2 * x - x0 - x1) / (x1 - x0))**2) * (k + 5) / 2 *
                      jacobi(k - 1, 3, 3)((2 * x - x0 - x1) / (x1 - x0))) / c


# Собственная функция на основе полученных в методе Ритца коэффициентов
def eig_vec_r(n, coef):
    return lambda x: sum([basic_func(i)(x) * coef[i] for i in range(n)])


# Метод минимальных невязок для поиска минимального СЗ и соответствующего СВ матрицы Грама
def scalar_product_method(Gamma_L, epsilon=1e-4):
    # Получаем размерность матрицы Грама
    n = Gamma_L.shape[0]
    # Генерируем случайный начальный вектор z
    z = np.random.rand(n)
    # Нормируем вектор z
    z /= np.linalg.norm(z)

    while True:
        # Решаем систему линейных уравнений для нахождения нового вектора
        z_new = np.linalg.solve(Gamma_L, z)

        # Нормируем новый вектор
        z_new_norm = np.linalg.norm(z_new)
        z_new /= z_new_norm

        if np.linalg.norm(z_new - z) < epsilon:
            break

        z = z_new

    # Вычисляем min СЗ
    lambda_min = np.dot(z, np.dot(Gamma_L, z)) / np.dot(z, z)

    return lambda_min, z


# Сравнение найденных минимальных СЗ с точным значением
def create_table(Nn, vals):
    df = pd.DataFrame()
    df['n'] = np.array(Nn)
    lambd_list = []
    lambd_loss = []

    for _ in Nn:
        G_l = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                G_l[i, j] = braces(basic_func(i), basic_func(j), dbasic_func(i), dbasic_func(j))

        lambd, coefs = scalar_product_method(G_l)
        lambd_list.append(lambd)
        lambd_loss.append(lambd - vals[0])
    df['lambda_1^n'] = lambd_list
    df['lambda_1^n-lambda_1^*'] = lambd_loss

    return df


if __name__ == '__main__':
    # Оценка функций p(x), q(x) (оцениваем диапазоны значений p(x) и q(x) на [x_0, x_1])
    x = np.linspace(x0, x1, 1000)

    # Вычисляем min и max функции p(x), построив массив значений p(x)
    p_min = np.min(p(np.linspace(x0, x1, 1000)))
    p_max = np.max(p(np.linspace(x0, x1, 1000)))

    # Вычисляем min и max функции q(x), построив массив значений q(x)
    q_min = np.min(q(np.linspace(x0, x1, 1000)))
    q_max = np.max(q(np.linspace(x0, x1, 1000)))

    for i in range(2):
        plt.plot(x, eigen_vec(i + 1)(x), label=f'Собственная функция {i + 1}')
    plt.title('Графики СФ для двух первых собственных значений')
    plt.legend()
    plt.show()

    # Таблица с результатами вычислений СЗ и их невязки для первых двух СФ при разных p и q
    df_mm = pd.DataFrame()

    df_mm['p'] = ['min', 'max']
    df_mm['lambda_1'] = [eigen_val(1, p_min, q_min), eigen_val(1, p_max, q_max)]
    df_mm['невязка 1'] = [loss(1, p_min, q_min, x), loss(1, p_max, q_max, x)]

    df_mm['lambda_2'] = [eigen_val(2, p_min, q_min), eigen_val(2, p_max, q_max)]
    df_mm['невязка 2'] = [loss(2, p_min, q_min, x), loss(2, p_max, q_max, x)]

    print(df_mm)

    # Собственные значения через "точные" собственные функции
    print("\n\nПриближенные значения через точные собственные функции")
    print(
        f'Первое собственное число '
        f'{braces(eigen_vec(1), eigen_vec(1), deigen_vec(1), deigen_vec(1)) / dotL2(eigen_vec(1), eigen_vec(1))}')
    print(
        f'Второе собственное число '
        f'{braces(eigen_vec(2), eigen_vec(2), deigen_vec(2), deigen_vec(2)) / dotL2(eigen_vec(2), eigen_vec(2))}')

    # Метод Ритца
    # Создаем матрицу Ритца и вычисляем ее элементы
    G_l = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            G_l[i, j] = braces(basic_func(i), basic_func(j), dbasic_func(i), dbasic_func(j))

    # Точные СЗ и СВ матрицы
    vals, vecs = eig(G_l)
    # Возвращаем индексы элементов массива vals в порядке возрастания
    sorted_indices = np.argsort(vals)
    # Упорядочиваем СЗ в порядке возрастания
    vals = vals[sorted_indices]
    # Упорядочиваем СВ в соответствии с отсортированными индексами
    # Каждый столбец матрицы vecs соответствует СВ и переупорядочивается так, чтобы они
    # соответствовали новому порядку СЗ
    vecs = vecs[:, sorted_indices]

    print("\n\nТочные собственные значения матрицы")
    print(f'Первое собственное число {vals[0]}')
    print(f'Второе собственное число {vals[1]}')

    for i in range(2):
        plt.plot(x, eig_vec_r(N, vecs[:, i])(x), label=f'Собственная функция {i + 1}')
    plt.title("Точные собственные функции")
    plt.legend()
    plt.show()

    # Задаем min СЗ матрицы Галеркина и соответствующий СВ
    lambda_min, coefs = scalar_product_method(G_l)

    plt.plot(x, eig_vec_r(N, coefs)(x), label='Собственная функция')
    plt.title("Собственная функция для метода Ритца")
    plt.legend()
    plt.show()

    result_table = create_table([3, 4, 5, 6, 7], vals)
    print(result_table)
