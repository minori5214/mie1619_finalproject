###########################################################################
# MIP with 'u' variables was significantly slower than the non-u version. #
###########################################################################


import tsplib95
import tsplib95.distances as distances

from gurobipy import *

from instance import Instance


instance = Instance('./instances/gr17.tsp')

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
u = m.addVars(N, vtype=GRB.INTEGER, lb=0, ub=N-1)

for j in range(N):
    m.addConstr(quicksum(x[i,j] for i in range(N) if i != j)==1)
for i in range(N):
    m.addConstr(quicksum(x[i,j] for j in range(N) if i != j)==1)

m.addConstr(u[0]==1)
for i in range(1, N):
    m.addConstr(2<= u[i])
    m.addConstr(u[i] <= N)
for i in range(1, N):
    for j in range(1, N):
        m.addConstr(u[j]>= u[i] + 1 - N*(1-x[i,j]))

#m.setParam("LogToConsole", 0)
m.setParam('TimeLimit', 300)

m.update() 
m.setObjective(quicksum(dist[i][j]*x[i,j] for i in range(N) for j in range(N)), GRB.MINIMIZE) ## constraint 4
m.update()

m.optimize()

optimal_cost = m.objVal
print("Current cost: ", optimal_cost)
print("Status: ", m.status)

for i in range(N):
    print([j for j in range(N) if x[i,j].X>0.0])
    #for j in range(N):
    #    if x[i,j].X > 0.0:
    #        print(j)