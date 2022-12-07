from itertools import combinations
from collections import namedtuple
import numpy as np

CAR_CAPACITY = 25
Problem = namedtuple('Problem', 'bikes parks cars D car_limits')


def read(n) -> Problem:
    with open(f'./data/input{n}.txt') as f:
        # количество самокатов, парковок и автомобилей
        line = f.readline()
        bikes, parks, cars = map(int, line.split())

        # матрица расстояний
        D = np.loadtxt(f, np.int, max_rows=bikes + parks + 1)

        # ограничения на длину маршрутов автомобилей
        line = f.readline()
        car_limits = list(map(int, line.split()))
    return Problem(bikes, parks, cars, D, car_limits)


def check(routes, bikes, parks, cars, D, car_limits) -> (int, str):
    if len(routes) != cars: return 0, 'Для каждой машины должен быть маршрут'

    for r in routes:
        if 0 in r: return 0, 'Выведенные маршруты не должны содержать точку 0.'
    for r1, r2 in combinations(routes, 2):  # type: list, list
        if set(r1) & set(r2): return 0, 'Выведенные маршруты не должны пересекаться по посещенным точкам'

    for r, car_limit in zip(routes, car_limits):
        length = sum(
            D[n1][n2]
            for n1, n2 in zip([0, ] + r, r)
        )
        if length > car_limit: return 0, 'Длина маршрута i-го автомобиля не должна превышать ограничения на длину'

    # Проверки наполнения
    for r in routes:
        bikes_in = 0
        for n in r:
            if n <= bikes:
                if bikes_in >= CAR_CAPACITY: return 0, 'При посещении точки с самокатом в автомобиле должно быть строго меньше 25 самокатов'
                bikes_in += 1
            else:
                if bikes_in <= 0: return 0, 'При посещении парковочного места в автомобиле должен быть хотя бы один самокат'
                bikes_in -= 1
        if bikes_in != 0: return 0, 'В конце маршрута в автомобиле не должно оставаться самокатов'

    score = sum(len(r) // 2 for r in routes)
    per_dist = score * 1000 / sum(car_limits)
    per_car = score / len(car_limits)
    return score, f'Самокатов перевезено: {score} из {min(bikes, parks)} ({per_dist:.1f}/км, {per_car:.1f}/авт.)'


def write(routes, n):
    with open(f'./output/output{n}.txt', 'w') as f:
        for route in routes:
            f.write(f'{len(route)} ' + ' '.join(map(str, route)) + '\n')


def check_output(n):
    res = []
    with open(f'./output/output{n}.txt', 'r') as f:
        for line in f:
            path = list(map(int, line.split()))
            res.append(path[1:])
    problem = read(n)
    return check(res, *problem)


def stats():
    bikes_total = 0
    for i in range(1, 31):
        bikes, parks, cars, D, car_limits = read(i)
        print(i, bikes, parks, cars)
        bikes_total += min(bikes, parks)
    print('Максимальный балл', bikes_total, '. Проходной бал 10000')


if __name__ == "__main__":
    stats()
