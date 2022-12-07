import numpy as np
from pycsp3 import *


def add_termination_node(D, inf):
    # Di,j = distance from i to j
    n = D.shape[0]
    D_new = np.zeros((n+1, n+1), dtype=D.dtype)
    D_new[:n, :n] = D
    D_new[-1, :n] = inf
    return D_new


def csp_solve(bikes, parks, cars, D, car_limits, capacity):
    D = add_termination_node(D, inf=max(car_limits))
    nodes = D.shape[0]
    start_node = 0
    end_node = nodes - 1

    o = VarArray(size=[cars, nodes], dom=range(nodes))
    c = VarArray(size=[cars, nodes], dom=range(0, capacity+1))
    D = cp_array(D.tolist())
    bike_cost = cp_array([0] + [1]*bikes + [-1]*parks + [0])

    satisfy(
        # start & end points
        [path[0] == start_node for path in o],
        # end points
        [path[-1] == end_node for path in o],
        # car is empty at the start
        [path[0] == 0 for path in c],
        # car is empty at in the end
        [path[-1] == 0 for path in c],

        # each node can be visited at most once
        Cardinality(o, occurrences={
            start_node: cars,
            **{n: range(0, 2) for n in range(start_node + 1, end_node)}
        }),

        # distance traveled by car
        [sum(D[i][j] for i, j in zip(o[car], o[car][1:])) <= car_limits[car] for car in range(cars)],

        # peeking & parking bike
        [
            [
                c[car][n] == c[car][n-1] + bike_cost[o[car][n]]
                for n in range(1, nodes)
            ]
            for car in range(cars)
        ]
    )

    maximize(
        # maximize nodes visited
        NValues(o)
    )
    if solve(verbose=2) is OPTIMUM:
        res = [
            [n for n in path if n != start_node and n != end_node]
            for path in values(o)
        ]
        return res


def manual(bikes, parks, cars, D, car_limits, capacity):
    import jinja2
    max_distance = max(car_limits)
    D = add_termination_node(D, inf=max_distance)
    nodes = D.shape[0]
    bike_costs = [0] + [1]*bikes + [-1]*parks + [0]

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(""),
    )
    template = env.get_template('problem.xml')
    problem = template.render(**locals())
    with open('../run2.xml', 'w', encoding='utf-8') as f:
        f.write(problem)

    # java -jar /home/boris/anaconda3/lib/python3.8/site-packages/pycsp3/solvers/ace/ACE-2.1.jar run.xml  -v=3

