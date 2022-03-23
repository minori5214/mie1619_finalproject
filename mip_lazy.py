import statistics
import tsplib95
import tsplib95.distances as distances

from gurobipy import *

from instance import Instance_MIP
from result import Result

class MIP():
    def __init__(self, instance_path, verbose=1):
        self.instance_path = instance_path
        self.instance = Instance_MIP(instance_path)
        self.verbose = verbose

        self.dim = self.instance.data.dimension
        self.dist = self.instance.dist

        self.build_complete = 0 # becomes 1 when the build is complete

    def build(self):
        self.model = Model()
        if self.verbose == 0:
            self.model.setParam("LogToConsole", 0)
        
        self.N = self.model.addVar(vtype=GRB.INTEGER)
        self.model.addConstr(self.N==self.dim)

        self.x = {}
        for i in range(self.dim):
            for j in range(self.dim):
                self.x[i,j] = self.model.addVar(vtype=GRB.BINARY, name="x({%s},{%s})" % (i,j))

        for j in range(self.dim):
            self.model.addConstr(quicksum(self.x[i,j] for i in range(self.dim) if i != j)==1)
        for i in range(self.dim):
            self.model.addConstr(quicksum(self.x[i,j] for j in range(self.dim) if i != j)==1)

        self.model._x = self.x
        self.model._N = self.N
        self.model.Params.lazyConstraints = 1

        self.model.update()
        self.model.setObjective(quicksum(self.dist[i][j]*self.x[i,j] for i in range(self.dim) for j in range(self.dim)), GRB.MINIMIZE) ## constraint 4

        self.build_complete = 1

    def solve(self, time_limit=None):
        assert self.build_complete == 1, \
            "Model is not initialized yet. Run MIP.build() first."
        
        if time_limit != None:
            self.model.setParam('TimeLimit', time_limit)

        self.model.update()
        self.model.optimize(subtourelim)

        objVal = self.model.objVal
        # Round to integer to prevent underflow
        if self.model.status == 2: objVal = round(self.model.objVal)
        
        statistics = {'NumConstrs': self.model.NumConstrs, 
                        'solve_time': self.model.Runtime, 
                        'status': self.model.Status}
 
        return Result(objVal, statistics)

    def resume(self, time_limit=None):
        return self.solve(time_limit=time_limit)

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        xs = model.cbGetSolution(model._x)
        N = int(model.cbGetSolution(model._N))
        selected = tuplelist((i, j) for i, j in model._x.keys()
                                if xs[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, N)
        if tour != None:
            model.cbLazy(quicksum(model._x[i,j] for i in tour for j in tour if i != j) <= len(tour)-1)
            #print("Lazy added!", tour)


# Given a tuplelist of edges, find the shortest subtour
def subtour(edges, N):
    unvisited = list(range(N))
    while unvisited:  # true if list is non-empty
        cycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) >= 2 and N-1 not in cycle:
            return cycle
    
    return None

if __name__ == "__main__":
    mip = MIP('./instances/bayg29.tsp')
    #print(mip.build)
    mip.build()
    #print("1st try")
    obj = mip.solve()
    print(obj)
    #print("2nd try")
    #mip.solve()
    #print("3rd try")
    #mip.solve()
    #print("4th try")
    #mip.solve()
    #print("5th try")
    #mip.solve()

    #for i in range(mip.N):
    #    for j in range(mip.N):
    #        if mip.x[i,j].X > 0.0:
    #            print(i, j)

"""
instance = Instance_MIP('./instances/xqf131.tsp')

# These we will use in our representation of a TSP problem: a list of
# (city, coord)-tuples.
#cities = [(city, tuple(coord)) for city, coord in data.node_coords.items()]
N = instance.data.dimension
dist = instance.dist

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        xs = model.cbGetSolution(model._x)
        selected = tuplelist((i, j) for i, j in model._x.keys()
                                if xs[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if tour != None:
            model.cbLazy(quicksum(x[i,j] for i in tour for j in tour if i != j) <= len(tour)-1)
            print("Lazy added!", tour)


# Given a tuplelist of edges, find the shortest subtour
def subtour(edges):
    unvisited = list(range(N))
    while unvisited:  # true if list is non-empty
        cycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) >= 2 and N-1 not in cycle:
            return cycle
    
    return None

m = Model()

x = {}
for i in range(N):
    for j in range(N):
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x({%s},{%s})" % (i,j))

for j in range(N):
    m.addConstr(quicksum(x[i,j] for i in range(N) if i != j)==1)
for i in range(N):
    m.addConstr(quicksum(x[i,j] for j in range(N) if i != j)==1)

m._x = x
m.Params.lazyConstraints = 1

m.update() 
m.setObjective(quicksum(dist[i][j]*x[i,j] for i in range(N) for j in range(N)), GRB.MINIMIZE) ## constraint 4
m.update()

m.optimize(subtourelim)

for i in range(N):
    for j in range(N):
        if x[i,j].X > 0.0:
            print(i, j)
"""