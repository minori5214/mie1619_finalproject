import tsplib95
import tsplib95.distances as distances

from docplex.cp.model import *
from docplex.cp.expression import *
from docplex.cp.parameters import CpoParameters

class Instance():
    def __init__(self, path):
        self.data = tsplib95.load(path)
        self.N = self.data.dimension

        self.dist = []
        if self.data.edge_weight_format == 'LOWER_DIAG_ROW':
            for i in range(self.N):
                tmp = [self.data.get_weight(i, j) for j in range(self.N)]
                tmp.append(self.data.get_weight(i, 0))
                self.dist.append(tmp)
        tmp = [self.data.get_weight(0, j) for j in range(self.N)]
        tmp.append(self.data.get_weight(0, 0))
        self.dist.append(tmp)

data = tsplib95.load('./instances/gr24.tsp')

instance = Instance('./instances/gr24.tsp')

# These we will use in our representation of a TSP problem: a list of
# (city, coord)-tuples.
#cities = [(city, tuple(coord)) for city, coord in data.node_coords.items()]
N = data.dimension
dist = instance.dist

solution = tsplib95.load('./instances/gr24.opt.tour')
tour = [[x-1 for x in solution.tours[0]]]
optimal = data.trace_tours(tour)[0]
print(tour)
sum = []
for i in range(N-1):
    sum.append(dist[tour[0][i]][tour[0][i+1]])
sum.append(dist[tour[0][-1]][tour[0][0]])
print(sum)
print('Total optimal tour length is {0}.'.format(optimal))

model = CpoModel(name='TSP')

#Model 1

x = [model.interval_var(name="x_{}".format(i), size=0) for i in range(N+1)]
seq = sequence_var(x, name="seq")

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

#Model 2

"""
x = [model.integer_var(min=1, max=N, name="x_{}".format(i)) for i in range(N)]

model.add(model.all_diff(x))
model.add(model.minimize(sum([x[i] for i in range(N)])))

msol= model.solve()
"""

print("Solution: ")
msol.print_solution()