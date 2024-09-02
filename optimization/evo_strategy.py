from operator import itemgetter
import random
import math

# константы
X_length = 2  # вектор варьируемых параметров
x_limit = 6  # ограничения функции
S_size = 1000  # число особей в популяции
P_mutation = 0.5  # вероятность мутации (шаг мутации)
t_max = 10000  # максимальное число итераций
max_fitness = -330  # оптимальное значение фитнесс функции
r = 10  # параметр отношения особей и потомков


# переменные
t = 0  # счетчик итераций
fitness_values = []  # значения фитнесс функции для особей популяции
max_fitness_values = [100]  # максимальные значения фитнесс функции
best_individual = [0, 0]  # особь с самым оптимальным значением фитнесс функции


# функции
def individual_fitness(individual):  # вычисление фитнесс функции
    '''x = individual[0]
    y = individual[1]
    return math.pow(math.pow(x, 2) + y - 11, 2) + math.pow(x + math.pow(y, 2) - 7, 2) - 330  # функция Химмельблау'''
    
    '''result = max_fitness
    for i in range(X_length):
        result += math.pow(individual[i], 2) - 10 * math.cos(2 * math.pi * individual[i]) + 10  # функция Растригина
    return result'''


def mutation(individual):  # оператор мутации
    children = []
    for j in range(r):
        parent = []
        for z in range(X_length):
            x = individual[z] + random.uniform(-P_mutation, P_mutation)
            while abs(x) > x_limit:
                x = individual[z] + random.uniform(-P_mutation, P_mutation)
            parent.append(x)
        parent.append(0)
        children.append(parent)
    return children


def individual_creator():  # создание особи
    individual = []
    for j in range(X_length):
        individual.append(random.uniform(-x_limit, x_limit))
    individual.append(0)
    return individual


def population_creator(n=0):  # создание популяции
    return list(individual_creator() for i in range(n))


S_population = population_creator(n=S_size)  # популяция


# главный цикл
while (t < t_max) and (min(max_fitness_values) > max_fitness) and (best_individual[1] < 250):

    between_population = []  # промежуточная популяция
    t += 1

    # добавляем в промежуточную популяцию родителей и потомков
    for individual_i in S_population:
        between_population.append(individual_i)
        children = mutation(individual_i)
        for k in range(r):
            between_population.append(children[k])

    # значения фитнесс функции промежуточной популяции
    fitness_values = list(map(individual_fitness, between_population))
    for individual_i, fitness_i in zip(between_population, fitness_values):
        individual_i[-1] = fitness_i

    # лучшее значение фитнесс функции
    max_fitness_values.append(min(fitness_values))

    # селекция лучших особей
    between_population.sort(key=itemgetter(X_length))
    S_population = between_population[:S_size]

    if S_population[0] != best_individual[0]:
        best_individual[0] = S_population[0]
        best_individual[1] = 0
    else:
        best_individual[1] += 1
    print(best_individual[0])
print('число итераций:' + str(t))
