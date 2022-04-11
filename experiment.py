import os

from pcost_min import Pcost_min
from bayesian import Bayesian
from nn import NN_Agent
from rnn import RNN_Agent
from onlineweightupdate import OnlineWeightUpdate

import numpy as np

ALGO_NUM = 3 # MIP, CP, ALNS

def experiment(method='pcost_min', write_header=False, save_to_csv=True,
                t=90, T=600, time_horizon=3):
    """
    T (int)           : total given time
    time_horizon (int): number of time steps that an algorithm is run for prediction
    t (int)           : (prediction algorithm) total given prediction time
    t (int)           : (switching algorithm) length of time of each time step
    delta_t (int)     : length of time of each time-step per algorithm
    
    """
    delta_t = t / time_horizon / ALGO_NUM

    if write_header and save_to_csv:
        if method != 'onlineweightupdate':
            with open('experiment_h6.csv', 'a') as f:
                f.write('Model, Instance, Solve Time, objVal, Status, Best Algo' + '\n')
        else:
            with open('experiment_switching.csv', 'a') as f:
                f.write('Model, Instance, Solve Time, objVal, Status, Weights' + '\n')

    if method == 'pcost_min':
        for file in os.listdir('test_instances'):
            print('file name: ', file)

            pcost = Pcost_min(t=t)
            objVal, solve_time, status, best_algo = pcost.run(os.path.join('test_instances', file))

            if save_to_csv:
                with open('experiment_h6.csv', 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'bayesian':
        bayes = Bayesian(delta_t=delta_t, t=t, T=T, time_horizon=time_horizon)
        X_train = np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        y_train = np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        bayes.fit(X_train, y_train)

        for file in os.listdir('test_instances'):
            print('file name: ', file)

            objVal, solve_time, status, best_algo = bayes.run(os.path.join('test_instances', file))

            if save_to_csv:
                with open('experiment_h6.csv', 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'nn':
        nn = NN_Agent(delta_t=delta_t, T=T, t=t, time_horizon=time_horizon, max_iter=500)
        X = np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        y = np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        nn.fit(X, y)

        for file in os.listdir('test_instances'):
            print('file name: ', file)

            objVal, solve_time, status, best_algo = nn.run(os.path.join('test_instances', file))

            if save_to_csv:
                with open('experiment_h6.csv', 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'rnn':
        rnn = RNN_Agent(delta_t=delta_t, T=T, time_horizon=time_horizon, max_iter=500)
        X = np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        y = np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        rnn.fit(X, y)

        for file in os.listdir('test_instances'):
            print('file name: ', file)

            objVal, solve_time, status, best_algo = rnn.run(os.path.join('test_instances', file))

            if save_to_csv:
                with open('experiment_h6.csv', 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'onlineweightupdate':
        algorithm = OnlineWeightUpdate(alpha=0.5, t=t, T=T)

        for file in os.listdir('test_instances'):
            print('file name: ', file)

            objVal, solve_time, status, weight_history = algorithm.run(os.path.join('test_instances', file))

            if save_to_csv:
                with open('experiment_switching.csv', 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + str(weight_history) + '\n'
                    )
            print(file, objVal, solve_time, status, weight_history)

if __name__ == "__main__":
    #experiment(method='pcost_min', write_header=True, t=90, delta_t=5, T=600, time_horizon=6)
    #experiment(method='bayesian', t=90, delta_t=5, T=600, time_horizon=6)
    #experiment(method='nn', t=90, delta_t=5, T=600, time_horizon=6)
    #experiment(method='rnn', t=90, delta_t=5, T=600, time_horizon=6)
    experiment(method='onlineweightupdate', write_header=True, t=60, T=600)
