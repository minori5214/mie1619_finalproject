from mip_lazy import MIP
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent
from result import Result

import numpy as np

class OnlineWeightUpdate():
    def __init__(self, algo_names=['MIP', 'CP', 'ALNS'], alpha=0.5, t=60, T=300):
        self.algo_names = algo_names
        self.alpha = alpha
        self.t = t
        self.T = T
    
    def run(self, instance_path):
        self.instance_path = instance_path

        self.base = {}
        self.base_id = {}
        k = 0
        if 'MIP' in self.algo_names:
            self.base['MIP'] = MIP(instance_path, verbose=0)
            self.base['MIP'].build()
            self.base_id[k] = 'MIP'
            k += 1
        if 'CP' in self.algo_names:
            self.base['CP'] = CP(instance_path, verbose=0)
            self.base['CP'].build()
            self.base_id[k] = 'CP'
            k += 1
        if 'ALNS' in self.algo_names:
            self.base['ALNS'] = ALNS_Agent(instance_path, verbose=0)
            self.base['ALNS'].build()
            self.base_id[k] = 'ALNS'
        
        weight = [1/len(self.base)]*len(self.base) # weights are set equal
        best_result = Result(float('inf'), None, None)
        num_iter = self.T // self.t
        solve_time = 0
        weight_history = [] # keep track of weights

        for i in range(num_iter):
            weight_history.append(weight)
            performance = []
            performance_prev = []
            for algo_id, algo in enumerate(self.base):
                if i == 0:
                    result = self.base[algo].solve(time_limit=self.t*weight[algo_id], tour=best_result.tour)
                else:
                    # From the second time, just resume optimization
                    result = self.base[algo].resume(time_limit=self.t*weight[algo_id], tour=best_result.tour)

                performance_prev.append(best_result.objVal)
                if best_result.objVal > result.objVal:
                    # If a better soluion is found, update the best result
                    best_result = result
                    performance.append(result.objVal)
                else:
                    performance.append(best_result.objVal)
                
                solve_time += result.statistics['solve_time']
                # Early stop if optimal
                if result.statistics['status'] == 2:
                    return best_result.objVal, solve_time, result.statistics['status'], weight_history
            
            if i == 0:
                try:
                    non_nan_maxima = max([x for x in performance if x != float('inf')])
                except:
                    non_nan_maxima = float('inf')
                #performance_prev = [x if x != float('inf') else non_nan_maxima for x in performance]
                #performance_prev = [max(performance)]*len(self.base)
            elif i >= 2: # Weights do not change in the first two iterations
                performance = [x if x != float('inf') else non_nan_maxima for x in performance]
            
                # add epsilon for preventing underflow
                cost_improvement = [(performance_prev[j]-performance[j])/(self.t*weight[j]) for j in range(len(self.base))]
                #print(performance_prev, performance)
                #print('raw cost improvement per sec', cost_improvement)

                # Normalize
                if sum(cost_improvement) != 0:
                    coeff = 1/sum(cost_improvement)
                    cost_improvement = [x*coeff for x in cost_improvement]
                else:
                    cost_improvement = [1/3, 1/3, 1/3] # all equal

                weight = [(1-self.alpha)*weight[j]+self.alpha*cost_improvement[j] for j in range(len(self.base))]
                assert abs(1.0 - sum(weight)) < 0.0001, \
                        "Invalid weight={}. performance_prev={}, \
                        performance={}, cost_improvement={}.".format(
                            weight, performance_prev, performance, cost_improvement)
                print(i, weight, best_result.objVal)

                performance_prev = performance
        
        return best_result.objVal, solve_time, result.statistics['status'], weight_history


if __name__ == "__main__":
    wupdate = OnlineWeightUpdate()
    wupdate.run('./instances/att532.tsp')