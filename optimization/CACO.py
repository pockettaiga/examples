from operator import itemgetter
import random
import math

# константы
ph_init = 1.0  # начальное кол-во феромона
ph_min = 0.00000001  # минимальное кол-во феромона
beta_ph = 0.1  # коэффициент устойчивости феромонных точек
P_mutation = 0.75  # вероятность мутации
step_max = 1  # максимальная величина шага мутации
beta_mutation = 0.5  # параметр нелинейности мутации
beta_beg = 1  # начальная величина шага локального поиска
beta_inc = 0.01  # инкремент шага локального поиска
alpha_0 = 0  # начальное значение возраста регионов
x_limit = 6  # ограничения функции
X_size = 2  # вектор варьируемых параметров, т.е. количество переменных
S_size = int(100 * (1 - math.exp(-0.1 * X_size)) + 5)  # число особей в популяции
t_max = 10000  # максимальное число итераций
fitness_end = -330  # глобальный минимум функции

# глобальные переменные
step_mutation = step_max  # шаг мутации
t = 1  # счетчик числа итераций
best_ant = [0] * (X_size + 2)  # муравей с лучшим значением фитнесс функции
repeat = 0


# функции
def fitness(z):  # вычисление фитнесс функции
    result = fitness_end
    '''for i in range(X_size):
        result += math.pow(z[i], 2) - 10 * math.cos(2 * math.pi * z[i]) + 10  # функция Растригина'''

    result += (z[0] ** 2 + z[1] - 11) ** 2 + (z[0] + z[1] ** 2 - 7) ** 2  # функция Химмельблау
    return result


def pheromone_reduce(d):  # испарение феромона
    if d >= ph_min:
        return beta_ph * d
    return 0


def dist(s1, s2):  # Евклидово расстояние между точками
    result = 0
    for k in range(X_size):
        result += (s1[k] - s2[k])**2
    return math.sqrt(result)


def dist_mean():  # среднее расстояние между двумя муравьями популяции
    n = 0
    result = 0
    for i in range(len(S_population) - 1):
        for j in range(i + 1, len(S_population)):
            result += dist(S_population[i], S_population[j])
            n += 1
    return result / n


def dot_interest(ant, ph):  # интерес муравья к данной феромонной точке
    r = dist(ant, ph)
    return (r_mean / 2) * ph[-1] * math.exp(-r)


def mass_centre(ant):  # координаты центра тяжести активных феромонных точек
    result = [0]*X_size
    x = 0
    for k in range(len(all_tracks)):
        x += dot_interest(ant, all_tracks[k])
    for i in range(len(all_tracks)):
        z = dot_interest(ant, all_tracks[i]) / x
        for j in range(X_size):
            result[j] += z * all_tracks[i][j]
    return result


def mutation(ant_global):  # оператор мутации
    child = []
    for i in range(X_size):
        x = ant_global[i] + random.uniform(-step_mutation, step_mutation)
        while abs(x) > x_limit:
            x = ant_global[i] + random.uniform(-step_mutation, step_mutation)
        child.append(x)
    child.append(0)
    child.append(alpha_0)
    return child


def step_reduce():  # изменение шага мутации
    return step_max * (1 - math.pow(random.uniform(0, 1), math.pow(1 - (t / t_max), beta_mutation)))


def crossover(s1, s2):  # оператор кроссовера
    child = []
    for i in range(X_size):
        x = s1[i] * random.uniform(0, 1) + s2[i] * random.uniform(0, 1)
        while abs(x) > x_limit:
            x = s1[i] * random.uniform(0, 1) + s2[i] * random.uniform(0, 1)
        child.append(x)
    child.append(0)
    child.append(alpha_0)
    return child


S_population = []  # регионы поиска
all_tracks = []  # феромонные следы

for i in range(S_size):  # создаем S_size регионов поиска
    value = [random.uniform(-x_limit, x_limit) for j in range(X_size)]

    S_population.append(value)
    S_population[i].append(0)  # здесь будет сохраняться значение фитнесс функции для данного региона
    S_population[i].append(alpha_0)  # weakness региона (начальное значение 0)

    all_tracks.append(S_population[i][:X_size])
    all_tracks[i].append(1)  # инициализируем в каждом из регионов феромонную точку


# главный цикл
while (t <= t_max) and (best_ant[X_size] != fitness_end) and (repeat < 350):

    for region in S_population:  # вычисление значения фитнесс функции
        if region[X_size] == 0:
            region[X_size] = fitness(region)
    S_population.sort(key=itemgetter(-2), reverse=True)  # сортируем в порядке возрастания значения фитнесс функции
    if S_population[-1] == best_ant:
        repeat += 1
    else:
        best_ant = S_population[-1]
        repeat = 0

    S_global = S_population[:int(0.9 * len(S_population))]  # муравьи для глобального поиска
    S_local = S_population[len(S_global):len(S_population)]  # муравьи для локального поиска

    # глобальный поиск
    S_crossover = S_global[int(0.9 * len(S_global)):]
    del S_global[int(0.9 * len(S_global)):]
    for i in range(len(S_global)):
        rand = random.uniform(0, 1)
        if rand <= P_mutation:
            S_global[i] = mutation(S_global[i])
            all_tracks.append(S_global[i][:X_size])
            all_tracks[-1].append(1)
    for i in range(len(S_crossover)):
        x = S_crossover[int(random.uniform(0, len(S_crossover)))]
        y = S_crossover[int(random.uniform(0, len(S_crossover)))]
        w = crossover(x, y)
        S_global.append(w)
        all_tracks.append(w[:X_size])
        all_tracks[-1].append(1)

    # локальный поиск лучшего региона
    S_local_i = S_local.pop(int(random.uniform(0, len(S_local))))
    r_mean = dist_mean()
    W = mass_centre(S_local_i)
    alpha = S_local_i[-1]
    S_local_new = [0]*(X_size + 2)
    for i in range(X_size):
        S_local_new[i] += S_local_i[i] + (W[i] - S_local_i[i]) * abs(beta_beg - beta_inc * alpha)
    S_local_i[X_size] = fitness(S_local_i)
    S_local_new[X_size] = fitness(S_local_new)
    if S_local_new[X_size] < S_local_i[X_size]:
        S_local_new[-1] += 0.5
        S_local.append(S_local_new)
    else:
        while (S_local_new[X_size] >= S_local_i[X_size]) and (alpha * beta_inc != beta_beg):
            alpha += 0.25
            S_local_better = [0]*(X_size + 2)
            for i in range(X_size):
                S_local_better[i] += S_local_i[i] + (W[i] - S_local_i[i]) * abs(beta_beg - beta_inc * alpha)
            S_local_better[X_size] = fitness(S_local_better)
            if S_local_better[X_size] < S_local_new[X_size]:
                alpha -= 0.25
                S_local_new = S_local_better
            if S_local_new[X_size] < S_local_i[X_size]:
                S_local_new[-1] += 0.5
                S_local.append(S_local_new)
                break
            if alpha * beta_inc == beta_beg:
                S_local.append(S_local_new)

    # испарение феромона
    for track in all_tracks:
        if track[-1] < ph_min:
            all_tracks.remove(track)
        else:
            track[-1] = pheromone_reduce(track[-1])

    S_population = S_global + S_local

    step_mutation = step_reduce()
    t += 1
    print(best_ant[:X_size], best_ant[X_size])
print('число итераций:' + str(t))
