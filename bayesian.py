from mip_lazy import MIP
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent

from sklearn.naive_bayes import GaussianNB
import numpy as np

class Bayesian():
    def __init__(self, algo_names=['MIP', 'CP', 'ALNS'], t=90, time_horizon=3, delta_t=10, T=300):
        """
        It solves the instance for "delta_t * time_horizon" [sec] by each method.
        Performance metric at each delta_t [sec] is recorded and then used for 
        determining the algorithm to use for the rest "T-delta_t*time_horizon*num(algorithms)" [sec]. 

        t (int)  : total given prediction time
        time_horizon (int): number of time steps that an algorithm is run for prediction

        
        """
        self.algo_names = algo_names
        self.t = t
        self.time_horizon = time_horizon
        self.T = T
        self.delta_t = delta_t

        self.model = GaussianNB()
    
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


        input_to_bayes = []
        for i in range(self.time_horizon):
            performance = []

            # For each algorithm, run for delta_t seconds
            for algo in self.base:
                if i == 0:
                    result = self.base[algo].solve(time_limit=self.delta_t)
                else:
                    # From the second time, just resume optimization
                    result = self.base[algo].resume(time_limit=self.delta_t)

                obj = result.objVal
                performance.append(obj)
            
            metric = [1 if x == min(performance) else 0 for x in performance] # best=1, behind=0
            input_to_bayes.extend(metric)
                
        # Choose the algorithm
        algo_id = self.model.predict([input_to_bayes])[0]
        best_algo = self.base_id[algo_id]
        #print(best_algo, "is selected")

        # Run the selected algorithm for the rest of the time
        remain = self.T-self.delta_t*self.time_horizon*len(self.base)
        result = self.base[best_algo].resume(remain)

        objVal = result.objVal
        if best_algo in ['MIP', 'CP']:
            solve_time = result.statistics['solve_time']
            status = result.statistics['status']
        elif best_algo == 'ALNS':
            solve_time = remain
            status = 9

        return objVal, solve_time+self.delta_t*self.time_horizon*len(self.base), status, best_algo
    
    def fit(self, X, y):
        assert X.shape[1] == len(self.algo_names) * self.time_horizon, \
            "Train data size does not match with the model size"

        self.model.fit(X, y)


if __name__ == "__main__":
    delta_t = 5
    T = 300
    time_horizon = 3
    bayes = Bayesian(delta_t=delta_t, T=T, time_horizon=time_horizon)
    X_train = np.load('X_t{}_T{}.npy'.format(delta_t, T))
    y_train = np.load('y_t{}_T{}.npy'.format(delta_t, T))

    bayes.fit(X_train, y_train)
    objVal = bayes.run('./instances/gr48.tsp')
    print(objVal)