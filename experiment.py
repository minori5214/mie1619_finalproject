import os
from time import time

from pcost_min import Pcost_min
from bayesian import Bayesian
from nn import NN_Agent
from rnn import RNN_Agent, RNN_Switching
from onlineweightupdate import OnlineWeightUpdate
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent

import numpy as np

ALGO_NUM = 3 # MIP, CP, ALNS

def experiment(method='pcost_min', write_header=False, save_to_csv=True,
                t=90, T=600, time_horizon=3, test_folder='test_instances_final', save_filename='experiment_final_h3'):
    """
    T (int)           : total given time
    time_horizon (int): number of time steps that an algorithm is run for prediction
    t (int)           : (prediction algorithm) total given prediction time
    t (int)           : (switching algorithm) length of time of each time step
    delta_t (int)     : length of time of each time-step per algorithm
    
    """
    delta_t = int(t / time_horizon / ALGO_NUM)
    print('delta_t: ', delta_t)

    if write_header and save_to_csv:
        if method != 'onlineweightupdate':
            with open('{}.csv'.format(save_filename), 'a') as f:
                f.write('Model, Instance, Solve Time, objVal, Status, Best Algo' + '\n')
        else:
            with open('{}_switching.csv'.format(save_filename), 'a') as f:
                f.write('Model, Instance, Solve Time, objVal, Status, Weights' + '\n')

    if method == 'cp':
        for file in os.listdir(test_folder):
            print('file name: ', file)

            cp = CP(os.path.join(test_folder, file), verbose=0)
            cp.build()
            result = cp.solve(time_limit=T)

            objVal = result.objVal
            status = result.statistics['status']
            solve_time = result.statistics['solve_time']
            best_algo = 'CP'

            if save_to_csv:
                with open('{}.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    if method == 'alns':
        for file in os.listdir(test_folder):
            print('file name: ', file)

            alns = ALNS_Agent(os.path.join(test_folder, file), verbose=0)
            alns.build()
            result = alns.solve(time_limit=T)

            objVal = result.objVal
            status = result.statistics['status']
            solve_time = result.statistics['solve_time']
            best_algo = 'ALNS'

            if save_to_csv:
                with open('{}.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    if method == 'pcost_min':
        for file in os.listdir(test_folder):
            print('file name: ', file)

            pcost = Pcost_min(t=t)
            objVal, solve_time, status, best_algo = pcost.run(os.path.join(test_folder, file))

            if save_to_csv:
                with open('{}.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'bayesian':
        bayes = Bayesian(delta_t=delta_t, t=t, T=T, time_horizon=time_horizon)
        X_train = np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        y_train = np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        bayes.fit(X_train, y_train)

        for file in os.listdir(test_folder):
            print('file name: ', file)

            objVal, solve_time, status, best_algo = bayes.run(os.path.join(test_folder, file))

            if save_to_csv:
                with open('{}.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'nn':
        nn = NN_Agent(delta_t=delta_t, t=t, T=T, time_horizon=time_horizon, max_iter=500)
        X = np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        y = np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        nn.fit(X, y)

        for file in os.listdir(test_folder):
            print('file name: ', file)

            objVal, solve_time, status, best_algo = nn.run(os.path.join(test_folder, file))

            if save_to_csv:
                with open('{}.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'rnn':
        rnn = RNN_Agent(delta_t=delta_t, t=t, T=T, time_horizon=time_horizon, max_iter=500)
        X = np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        y = np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
        rnn.fit(X, y)

        for file in os.listdir(test_folder):
            print('file name: ', file)

            objVal, solve_time, status, best_algo = rnn.run(os.path.join(test_folder, file))

            if save_to_csv:
                with open('{}.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + best_algo + '\n'
                    )
            print(file, objVal, solve_time, status, best_algo)

    elif method == 'onlineweightupdate':
        algorithm = OnlineWeightUpdate(alpha=0.5, t=t, T=T)

        for file in os.listdir(test_folder):
            print('file name: ', file)

            objVal, solve_time, status, weight_history = algorithm.run(os.path.join(test_folder, file))

            if save_to_csv:
                with open('{}_switching.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + str(weight_history) + '\n'
                    )
            print(file, objVal, solve_time, status, weight_history)

    elif method == 'rnn_switching':
        rnn = RNN_Switching(alpha=0.5, t=t*ALGO_NUM, T=T, max_iter=10000)
        X = np.load('X_rnn_t{}_T{}_h{}.npy'.format(t, T, time_horizon))
        y = np.load('y_rnn_t{}_T{}_h{}.npy'.format(t, T, time_horizon))
        rnn.fit(X, y)

        for file in os.listdir(test_folder):
            print('file name: ', file)

            objVal, solve_time, status, weight_history = rnn.run(os.path.join(test_folder, file))

            if save_to_csv:
                with open('{}_switching.csv'.format(save_filename), 'a') as f:
                    f.write(method + ',' + file.split('.')[0] + ',' + str(solve_time) + ','
                            + str(objVal) + ',' + str(status) + ',' + str(weight_history) + '\n'
                    )
            print(file, objVal, solve_time, status, weight_history)

if __name__ == "__main__":
    experiment(method='alns', write_header=True, t=90, T=600, test_folder='test_instances_final', save_filename='additional_experiment')
    experiment(method='alns', write_header=True, t=45, T=300, test_folder='test_instances', save_filename='additional_experiment')
    #experiment(method='pcost_min', write_header=True, t=45, T=300, test_folder='test_instances_final', save_filename='experiment_final_h3_t300')
    #experiment(method='bayesian', t=45, T=300, time_horizon=3, test_folder='test_instances_final', save_filename='experiment_final_h3_t300')
    #experiment(method='nn', t=45, T=300, time_horizon=3, test_folder='test_instances_final', save_filename='experiment_final_h3_t300')
    #experiment(method='rnn', t=45, T=300, time_horizon=3, test_folder='test_instances_final', save_filename='experiment_final_h3_t300')
    #experiment(method='onlineweightupdate', write_header=True, t=60, T=300, test_folder='test_instances_final', save_filename='experiment_final_h3_t300')
    #experiment(method='rnn_switching', t=20, T=300, test_folder='test_instances_final', save_filename='experiment_final_h3_t300')
