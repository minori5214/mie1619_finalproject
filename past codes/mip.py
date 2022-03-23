import tsplib95
import tsplib95.distances as distances

from gurobipy import *

from instance import Instance_MIP

#data = tsplib95.load('./instances/gr24.tsp')

instance = Instance_MIP('./instances/gr17.tsp')

# These we will use in our representation of a TSP problem: a list of
# (city, coord)-tuples.
#cities = [(city, tuple(coord)) for city, coord in data.node_coords.items()]
N = instance.data.dimension
dist = []
for i in range(N):
    tmp = [instance.data.get_weight(i, j) for j in range(N)]
    dist.append(tmp)
tmp = [instance.data.get_weight(0, j) for j in range(N)]
dist.append(tmp)

m = Model()

x = {}
for i in range(N):
    for j in range(N):
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x({%s},{%s})" % (i,j))

for j in range(N):
    m.addConstr(quicksum(x[i,j] for i in range(N) if i != j)==1)
for i in range(N):
    m.addConstr(quicksum(x[i,j] for j in range(N) if i != j)==1)

import itertools
nodes = [x for x in range(N)]
S_all = []
for i in range(2, (N-2)+1):
    print(i)
    for c in itertools.combinations(nodes, i):
        #S_all.append(list(c))
        m.addConstr(quicksum(x[i,j] for i in c for j in c if i != j) <= len(c)-1)
#for S in S_all:
#    nonS = [x for x in range(N) if x not in S]
#    print(S, nonS)
#    #m.addConstr(quicksum(x[i,j] for j in nonS for i in S) >= 1)

m.update() 
m.setObjective(quicksum(dist[i][j]*x[i,j] for i in range(N) for j in range(N)), GRB.MINIMIZE) ## constraint 4
m.update()

m.optimize()

for i in range(N):
    for j in range(N):
        if x[i,j].X > 0.0:
            print(j)