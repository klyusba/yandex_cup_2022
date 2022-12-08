from enum import Enum
import pickle
from datetime import datetime

from ortools.constraint_solver import routing_enums_pb2 as enums
from ortools.constraint_solver import pywrapcp as cp
from ortools.init import pywrapinit
import numpy as np


class ConstraintType(Enum):
    HARD = 'hard'
    SOFT = 'soft'
    NONE = 'none'


class Model:
    COEF_CAPACITY = 100_000
    COEF_DISTANCE = 10_000
    COEF_NODE = 1_000_000

    _on_solution_callbacks = []

    default_params = {
        "first_solution_strategy": enums.FirstSolutionStrategy.AUTOMATIC,
        "use_unfiltered_first_solution_strategy": False,
        "savings_neighbors_ratio": 1,
        "savings_max_memory_usage_bytes": 6_000_000_000,
        "savings_add_reverse_arcs": False,
        "savings_arc_coefficient": 1,
        "savings_parallel_routes": False,
        "cheapest_insertion_farthest_seeds_ratio": 0,
        "cheapest_insertion_first_solution_neighbors_ratio": 1,
        "cheapest_insertion_ls_operator_neighbors_ratio": 1,
        "local_search_operators": {
            "use_relocate": cp.BOOL_TRUE,
            "use_relocate_pair": cp.BOOL_TRUE,
            "use_light_relocate_pair": cp.BOOL_TRUE,
            "use_relocate_subtrip": cp.BOOL_TRUE,
            "use_relocate_neighbors": cp.BOOL_FALSE,
            "use_exchange": cp.BOOL_TRUE,
            "use_exchange_pair": cp.BOOL_TRUE,
            "use_exchange_subtrip": cp.BOOL_TRUE,
            "use_cross": cp.BOOL_TRUE,
            "use_cross_exchange": cp.BOOL_FALSE,
            "use_relocate_expensive_chain": cp.BOOL_TRUE,
            "use_two_opt": cp.BOOL_TRUE,
            "use_or_opt": cp.BOOL_TRUE,
            "use_lin_kernighan": cp.BOOL_TRUE,
            "use_tsp_opt": cp.BOOL_FALSE,
            "use_make_active": cp.BOOL_TRUE,
            "use_relocate_and_make_active": cp.BOOL_FALSE,
            "use_make_inactive": cp.BOOL_TRUE,
            "use_make_chain_inactive": cp.BOOL_FALSE,
            "use_swap_active": cp.BOOL_TRUE,
            "use_extended_swap_active": cp.BOOL_FALSE,
            "use_node_pair_swap_active": cp.BOOL_TRUE,
            "use_path_lns": cp.BOOL_FALSE,
            "use_full_path_lns": cp.BOOL_FALSE,
            "use_tsp_lns": cp.BOOL_FALSE,
            "use_inactive_lns": cp.BOOL_FALSE,
            "use_global_cheapest_insertion_path_lns": cp.BOOL_TRUE,
            "use_local_cheapest_insertion_path_lns": cp.BOOL_TRUE,
            "use_global_cheapest_insertion_expensive_chain_lns": cp.BOOL_FALSE,
            "use_local_cheapest_insertion_expensive_chain_lns": cp.BOOL_FALSE,
        },
        "relocate_expensive_chain_num_arcs_to_consider": 4,
        "heuristic_expensive_chain_lns_num_arcs_to_consider": 4,
        "local_search_metaheuristic": enums.LocalSearchMetaheuristic.AUTOMATIC,
        "guided_local_search_lambda_coefficient": 0.1,
        "use_depth_first_search": False,
        "use_cp": cp.BOOL_TRUE,
        "use_cp_sat": cp.BOOL_FALSE,
        # "continuous_scheduling_solver": GLOP,
        # "mixed_integer_scheduling_solver": CP_SAT,
        "optimization_step": 0.0,
        "number_of_solutions_to_collect": 1,
        "time_limit": 0x7fffffffffffffff,
        "solution_limit": 0x7fffffffffffffff,
        "lns_time_limit_ms": 100,
        "use_full_propagation": False,
        "log_search": False,
    }

    def __init__(self, name, distance: ConstraintType = 'soft', capacity: ConstraintType = 'soft', start_node=0, verbose=0, **kwargs):
        self.name = name
        self._distance_type = distance
        self._capacity_type = capacity
        self._start_node = start_node
        self._verbose = verbose
        self._params = {**self.default_params, **kwargs, "log_search": verbose == 3}

        self._manager = None
        self._routing = None

    def _compile(self, D, bike_cost, car_limits, capacity):
        self._D, self._bike_cost, self._car_limits = D, bike_cost, car_limits

        nodes = D.shape[0]
        cars = len(car_limits)
        # Create the routing index manager.
        self._manager = cp.RoutingIndexManager(nodes, cars, self._start_node)

        # Create Routing Model.
        self._routing = cp.RoutingModel(self._manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = self._manager.IndexToNode(from_index)
            to_node = self._manager.IndexToNode(to_index)
            return D[from_node, to_node]

        distance_callback_index = self._routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        self._routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        if self._distance_type == ConstraintType.HARD:
            self._distance_hard(distance_callback_index, car_limits)
        elif self._distance_type == ConstraintType.SOFT:
            self._distance_soft(distance_callback_index, car_limits)

        def capacity_callback(from_index):
            from_node = self._manager.IndexToNode(from_index)
            return bike_cost[from_node]

        capacity_callback_index = self._routing.RegisterUnaryTransitCallback(capacity_callback)
        if self._capacity_type == ConstraintType.HARD:
            self._capacity_hard(capacity_callback_index, capacity)
        elif self._capacity_type == ConstraintType.SOFT:
            self._capacity_soft(capacity_callback_index, capacity)

        # Allow to drop nodes.
        for node in range(1, nodes):
            self._routing.AddDisjunction([self._manager.NodeToIndex(node)], self.COEF_NODE)

        return self

    def _distance_soft(self, callback, car_limits):
        self._routing.AddDimensionWithVehicleCapacity(
            callback,
            0,  # no slack
            [5 * lim for lim in car_limits],  # vehicle maximum travel distance
            True,  # start cumul to zero
            'Distance'
        )

        d = self._routing.GetDimensionOrDie('Distance')
        for car, limit in enumerate(car_limits):
            index = self._routing.End(car)
            d.SetCumulVarSoftUpperBound(index, limit, self.COEF_DISTANCE)

    def _distance_hard(self, callback, car_limits):
        self._routing.AddDimensionWithVehicleCapacity(
            callback,
            0,  # no slack
            car_limits,  # vehicle maximum travel distance
            True,  # start cumul to zero
            'Distance'
        )

    def _capacity_soft(self, callback, capacity):
        nodes, cars = self._routing.nodes(), self._routing.vehicles()
        self._routing.AddDimensionWithVehicleCapacity(
            callback,
            0,  # no slack
            [nodes] * cars,
            False,  # start cumul to zero
            'Capacity'
        )
        d = self._routing.GetDimensionOrDie('Capacity')
        for n in range(nodes):
            index = self._manager.NodeToIndex(n)
            d.CumulVar(index).SetMin(-nodes)
            d.SetCumulVarSoftLowerBound(index, 0, self.COEF_CAPACITY)
            d.SetCumulVarSoftUpperBound(index, capacity, self.COEF_CAPACITY)

        for car in range(cars):
            end_index = self._routing.End(car)
            d.SetCumulVarSoftUpperBound(end_index, 0, self.COEF_NODE)  # they will be discarded

    def _capacity_hard(self, callback, capacity):
        solver, cars = self._routing.solver(), self._routing.vehicles()
        self._routing.AddDimensionWithVehicleCapacity(
            callback,
            0,  # no slack
            [capacity] * cars,
            True,  # start cumul to zero
            'Capacity'
        )

        d = self._routing.GetDimensionOrDie('Capacity')
        for car in range(cars):
            end_index = self._routing.End(car)
            solver.Add(d.CumulVar(end_index) == 0)

    def _parse_solution(self, solution):
        """Prints solution on console."""
        res = []
        for car in range(self._routing.vehicles()):
            path = []
            index = self._routing.Start(car)
            plan_output = 'Route for vehicle {}:'.format(car)
            route_distance = 0
            bikes_in = 0
            while not self._routing.IsEnd(index):
                n = self._manager.IndexToNode(index)
                bikes_in += self._bike_cost[n]
                plan_output += f' {n}|{bikes_in} -> '
                path.append(n)
                index = solution.Value(self._routing.NextVar(index))
                n2 = self._manager.IndexToNode(index)
                route_distance += self._D[n, n2]

            n = self._manager.IndexToNode(index)
            bikes_in += self._bike_cost[n]
            path.append(n)
            plan_output += f'{n}|{bikes_in}\n'
            plan_output += 'Distance of the route: {}/{}'.format(route_distance, self._car_limits[car])
            if self._verbose >= 1:
                print(plan_output)
            res.append(path)

        if self._verbose >= 1:
            score = sum(sum(max(self._bike_cost[n], 0) for n in r) - max(sum(self._bike_cost[n] for n in r), 0) for r in res)
            print(f'{self.name}: Objective: {solution.ObjectiveValue()}, bikes: {score}')
        return res

    def _monitor_for_early_stop(self, failure_limit: int = 2**63) -> callable:
        # FIXME memory leak
        class RoutingMonitor:
            def __init__(self, model: cp.RoutingModel):
                self.model = model
                self._counter = 0
                self._best_objective = 2 ** 63
                self._counter_limit = failure_limit

            def __call__(self):
                obj = self.model.CostVar().Max()
                if self._best_objective > obj:
                    self._best_objective = obj
                    self._counter = 0
                else:
                    self._counter += 1
                    if self._counter > self._counter_limit:
                        print('limit reached')
                        self.model.solver().FinishCurrentSearch()

        return RoutingMonitor(self._routing)

    def _gather_variables(self):
        obj = self._routing.CostVar().Max()

        res = []
        for car in range(self._routing.vehicles()):
            path = []
            index = self._routing.Start(car)
            while not self._routing.IsEnd(index):
                n = self._manager.IndexToNode(index)
                path.append(n)
                index = self._routing.NextVar(index).Value()

            n = self._manager.IndexToNode(index)
            path.append(n)
            res.append(path)
        return obj, res

    @classmethod
    def add_monitor(cls, func: callable):
        cls._on_solution_callbacks.append(func)

    def _monitor_callbacks(self):
        obj, paths = self._gather_variables()
        paths = [path[1:-2] for path in paths]
        for f in self._on_solution_callbacks:
            f(obj, paths)

    def _make_params(self, **kwargs):
        # TODO
        params = {**self._params, **kwargs}
        search_parameters = cp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = enums.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
        search_parameters.local_search_metaheuristic = enums.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.use_full_propagation = params['use_full_propagation']
        if params['use_operators']:
            search_parameters.relocate_expensive_chain_num_arcs_to_consider = 10
            search_parameters.heuristic_expensive_chain_lns_num_arcs_to_consider = 10
            search_parameters.local_search_operators.use_make_chain_inactive = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_extended_swap_active = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_node_pair_swap_active = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_tsp_opt = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_relocate_neighbors = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_cross_exchange = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_path_lns = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_full_path_lns = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_tsp_lns = cp.BOOL_TRUE
            search_parameters.local_search_operators.use_inactive_lns = cp.BOOL_TRUE
        search_parameters.log_search = params['log_search']
        return search_parameters

    def solve(
        self,
        D,
        bike_cost,
        car_limits,
        capacity,
        *,
        initial_solution=None,
        use_full_propagation=False,
        time_limit=180,
        lns_time_limit_ms=10,
        failure_limit=10,
        use_operators=False
    ):
        self._compile(D, bike_cost, car_limits, capacity)

        # self._routing.AddAtSolutionCallback(self._monitor_for_early_stop(failure_limit))
        if self._on_solution_callbacks:
            self._routing.AddAtSolutionCallback(self._monitor_callbacks)

        search_parameters = self._make_params(
            use_operators=use_operators,
            use_full_propagation=use_full_propagation
        )
        search_parameters.time_limit.seconds = time_limit
        if lns_time_limit_ms >= 1000:
            search_parameters.lns_time_limit.seconds = lns_time_limit_ms // 1_000
        else:
            search_parameters.lns_time_limit.nanos = int(lns_time_limit_ms * 1_000_000)

        if initial_solution:
            self._routing.CloseModelWithParameters(search_parameters)
            sol = [path[1:-1] for path in initial_solution]  # remove depot
            sol = self._routing.ReadAssignmentFromRoutes(sol, True)
            assert sol is not None
            solution = self._routing.SolveFromAssignmentWithParameters(sol, search_parameters)
        else:
            solution = self._routing.SolveWithParameters(search_parameters)

        return self._parse_solution(solution)

    def save_solution(self, paths, filename=None):
        if filename is None:
            filename = self.name + '_solution.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_solution(self, filename=None):
        if filename is None:
            filename = self.name + '_solution.pickle'
        with open(filename, 'rb') as f:
            return pickle.load(f)


def _init_log():
    cpp_flags = pywrapinit.CppFlags()
    cpp_flags.logtostderr = True
    cpp_flags.log_prefix = False
    pywrapinit.CppBridge.SetFlags(cpp_flags)
    pywrapinit.CppBridge.InitLogging(__name__)


def add_termination_nodes(D, bike_cost, cars, inf):
    n = D.shape[0]
    D_new = np.zeros((n + cars, n + cars), dtype=D.dtype)
    D_new[:n, :n] = D
    D_new[:n, 0] = inf
    D_new[n:, n:] = inf
    D_new[-cars:, 1:n] = inf
    D_new -= np.diag(np.diag(D_new, 0))

    bike_cost = bike_cost + [0, ] * cars
    return D_new, bike_cost


def filter_path(paths, D, bike_cost, car_limits, capacity):
    res = []
    for path, limit in zip(paths, car_limits):
        c, d, p = 0, 0, []
        n1 = 0
        for n in path:
            arc, n1 = D[n1, n], n
            d += arc
            if d >= limit:
                p += path[-2:]
                break

            b = bike_cost[n]
            if b == 1 and c < capacity:
                p.append(n)
                c += 1
            elif b == -1 and c > 0:
                p.append(n)
                c -= 1
            elif b == 0:
                p.append(n)

        i = len(p) - 1
        while c > 0:
            if bike_cost[p[i]] == 1:
                p.pop(i)
                c -= 1
            i -= 1
        res.append(p)

    return res


def solve(bikes, parks, cars, D, car_limits, capacity, name='Model', verbose=0):
    bike_cost = [0] + [1] * bikes + [-1] * parks
    D, bike_cost = add_termination_nodes(D, bike_cost, cars, inf=100 * max(car_limits))
    P = D, bike_cost, car_limits, capacity

    def log_obj(obj, paths):
        print(f'{datetime.now():%H:%M:%S} {obj:_}')
    if verbose == 2:
        Model.add_monitor(log_obj)
    elif verbose == 3:
        _init_log()

    if verbose > 0:
        print(f'\n{name}: Step without capacity', flush=True)
    prev_solution, solution = None, None
    m = Model(
        name,
        capacity=ConstraintType.NONE,
        distance=ConstraintType.HARD,
        verbose=verbose
    )
    solution = m.solve(*P, initial_solution=solution, time_limit=600)

    m = Model(
        name,
        capacity=ConstraintType.SOFT,
        distance=ConstraintType.HARD,
        verbose=verbose
    )
    for i in range(30):
        if verbose > 0:
            print(f'\n{name}: Step {i} with soft capacity', flush=True)
        solution = m.solve(*P,
                           initial_solution=solution,
                           time_limit=120,
                           use_full_propagation=i % 2 == 1,
                           use_operators=True)
        if solution == prev_solution:
            break
        prev_solution = solution
    solution = filter_path(solution, D, bike_cost, car_limits, capacity)
    prev_solution = solution

    m = Model(
        name,
        capacity=ConstraintType.HARD,
        distance=ConstraintType.HARD,
        verbose=verbose
    )
    for i in range(30):
        if verbose > 0:
            print(f'\n{name}: All hard step {i}', flush=True)
        lns_time_limit_ms = 1000 if i >= 5 else 100
        solution = m.solve(*P,
                           initial_solution=solution,
                           time_limit=120,
                           lns_time_limit_ms=lns_time_limit_ms,
                           use_full_propagation=i % 2 == 1,
                           use_operators=True)
        if solution == prev_solution:
            break
        prev_solution = solution
    return [path[1:-2] for path in solution]
