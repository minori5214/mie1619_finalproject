import tsplib95
import tsplib95.distances as distances

from docplex.cp.model import *
from docplex.cp.expression import *
from docplex.cp.parameters import CpoParameters

from instance import Instance_CP
from result import Result

class CP():
    def __init__(self, instance_path, verbose=1):
        self.instance_path = instance_path
        self.instance = Instance_CP(instance_path)
        self.verbose = verbose

        self.N = self.instance.data.dimension
        self.dist = self.instance.dist

        self.build_complete = 0
    
    def build(self):
        self.model = CpoModel(name='TSP')

        #Model 1

        self.x = [self.model.interval_var(name="x_{}".format(i), size=0) for i in range(self.N+1)]
        self.seq = sequence_var(self.x, name="seq")

        self.model.add(self.model.no_overlap(self.seq,  distance_matrix=self.dist, is_direct=True))
        self.model.add(self.model.first(self.seq, self.x[0]))
        self.model.add(self.model.last(self.seq, self.x[self.N]))

        self.model.add(self.model.minimize(max([end_of(self.x[i]) for i in range(self.N+1)])))

        self.build_complete = 1
    
    def solve(self, time_limit=None):
        assert self.build_complete == 1, \
            "Model is not initialized yet. Run MIP.build() first."

        self.params = CpoParameters()
        if time_limit != None:
            self.params.set_TimeLimit(time_limit)


        if self.verbose == 1:
            self.msol= self.model.solve(params=self.params)
        else:
            self.msol= self.model.solve(params=self.params, log_output=None)

        #optimal_cost = self.msol.get_objective_value()
        #optimal_x = [self.msol.get_value('x_{}'.format(i)).start for i in range(self.N)]
        #optimal_seq = self.msol.get_value('seq')
        #[msol.get_value('x_{}'.format(j)).start for j in range(n)]

        status = self.msol.get_solve_status()
        if status == 'Optimal': status = 2
        elif status == 'Feasible': status = 9

        statistics = {'NumConstrs': CpoModelStatistics(self.model).get_number_of_constraints(), 
                        'solve_time': self.msol.get_solve_time(), 
                        'status': status}

        return Result(self.msol.get_objective_value(), statistics)
    
    def resume(self, time_limit=30):
        if self.msol == None:
            return self.msol.get_objective_value()

        cur_xs = [self.msol.get_value('x_{}'.format(i)) for i in range(self.N)]

        self.new_model = CpoModel()
        self.new_x = [self.new_model.interval_var(name="x_{}".format(i), size=0) for i in range(self.N+1)]
        self.new_seq = sequence_var(self.new_x, name="seq")

        self.new_model.add(self.new_model.no_overlap(self.new_seq,  distance_matrix=self.dist, is_direct=True))
        self.new_model.add(self.new_model.first(self.new_seq, self.new_x[0]))
        self.new_model.add(self.new_model.last(self.new_seq, self.new_x[self.N]))

        self.new_model.add(self.new_model.minimize(max([end_of(self.new_x[i]) for i in range(self.N+1)])))

        stp = self.new_model.create_empty_solution()
        for i in range(self.N):
            stp.add_interval_var_solution(
                self.new_x[i], start=cur_xs[i].start, end=cur_xs[i].end, size=cur_xs[i].size
            )
        self.new_model.set_starting_point(stp)


        self.model = self.new_model
        return self.solve(time_limit=time_limit)


if __name__ == "__main__":
    cp = CP('./instances/bays29.tsp')
    cp.build()
    obj = cp.solve()
    print(obj)
    obj = cp.resume()
    print(obj)

    """
    instance = Instance_CP('./instances/ulysses16.tsp')

    # These we will use in our representation of a TSP problem: a list of
    # (city, coord)-tuples.
    #cities = [(city, tuple(coord)) for city, coord in data.node_coords.items()]
    N = instance.N
    dist = instance.dist

    #solution = tsplib95.load('./instances/xqf131.opt.tour')
    #optimal = instance.data.trace_tours(solution.tours)[0]

    #solution = tsplib95.load('./instances/xqf131.opt.tour')
    #tour = [[x-1 for x in solution.tours[0]]]
    #optimal = instance.data.trace_tours(tour)[0]

    model = CpoModel(name='TSP')

    #Model 1

    x = [model.interval_var(name="x_{}".format(i), size=0) for i in range(N+1)]
    seq = sequence_var(x, name="seq")

    print(dist)
    model.add(model.no_overlap(seq,  distance_matrix=dist, is_direct=True))
    model.add(model.first(seq, x[0]))
    model.add(model.last(seq, x[N]))

    model.add(model.minimize(max([end_of(x[i]) for i in range(N+1)])))

    params = CpoParameters()
    params.set_TimeLimit(30)

    msol= model.solve(params=params, log_output=None)

    print(msol.get_solve_status())
    #if not model.solve():
    #    model.refine_conflict().write()

    optimal_cost = msol.get_objective_value()
    optimal_x = [msol.get_value('x_{}'.format(i)).start for i in range(N)]
    print("Optimal cost: ", optimal_cost)
    print("Optimal x", optimal_x)

    optimal_seq = msol.get_value('seq')
    msol.print_solution()

    #Model 2

    #x = [model.integer_var(min=1, max=N, name="x_{}".format(i)) for i in range(N)]

    #model.add(model.all_diff(x))
    #model.add(model.minimize(sum([x[i] for i in range(N)])))

    #msol= model.solve()

    print("Solution: ")
    msol.print_solution()
    """