from utils import *
from solvers.or_solver import solve
from multiprocessing import Pool

# https://contest.yandex.ru/yacup/contest/42200/download/A/


def _solve(problem_num, verbose=1):
    problem = read(problem_num)
    res = solve(*problem, name=str(problem_num), verbose=verbose)
    score, msg = check(res, *problem)
    write(res, problem_num)
    print(problem_num, msg, flush=True)
    return score


def main(n_jobs=8):
    p = Pool(n_jobs, maxtasksperchild=1)
    scores = p.map(_solve, range(1, 31))
    total = sum(scores)
    print(f'{total=}')


def score():
    scores = list(map(check_output, range(1, 31)))
    total = sum(bikes for bikes, _ in scores)
    print(f'{total=}')
    for i, (_, msg) in enumerate(scores, start=1):
        print(i, msg)


if __name__ == "__main__":
    main(n_jobs=4)
    # _solve(1, verbose=3)
    # score()
