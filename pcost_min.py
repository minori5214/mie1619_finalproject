import tsplib95
from mip_lazy import MIP
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent


class Pcost_min():
    def __init__(self, algo_names=['MIP', 'CP', 'ALNS'], t=15, T=300):
        """
        t (int): total given prediction time
        T (int): total given time time

        """
        
        self.algo_names = algo_names
        self.t = t
        self.T = T
    
    def run(self, instance_path):
        self.instance_path = instance_path

        self.base = {}
        if 'MIP' in self.algo_names:
            self.base['MIP'] = MIP(instance_path, verbose=0)
        if 'CP' in self.algo_names:
            self.base['CP'] = CP(instance_path, verbose=0)
        if 'ALNS' in self.algo_names:
            self.base['ALNS'] = ALNS_Agent(instance_path, verbose=0)

        pcost = {}

        # For each algorithm, run for t seconds
        for algo in self.base:
            self.base[algo].build()
            result = self.base[algo].solve(time_limit=self.t // len(self.base))
            pcost[algo] = result.objVal

        # Choose the algorithm
        best_algo = min(pcost, key=pcost.get)
        #print(pcost)
        #print(best_algo, "is selected")

        # Run the selected algorithm for T-t*num(algo) seconds
        remain = self.T-self.t
        result = self.base[best_algo].resume(time_limit=remain)

        objVal = result.objVal
        if best_algo in ['MIP', 'CP']:
            solve_time = result.statistics['solve_time']
            status = result.statistics['status']
        elif best_algo == 'ALNS':
            solve_time = remain
            status = 9

        return objVal, solve_time+self.t, status, best_algo


if __name__ == "__main__":
    pcost = Pcost_min()
    objVal = pcost.run('./instances/gr48.tsp')
    print(objVal)


