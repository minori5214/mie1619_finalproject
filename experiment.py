import os

from pcost_min import Pcost_min
from bayesian import Bayesian
from nn import NN_Agent
from rnn import RNN_Agent

import numpy as np

def experiment(method='pcost_min', write_header=False, save_to_csv=True,
                t=15, delta_t=5, T=300, time_horizon=3):
    assert t == delta_t * time_horizon, "Time does not match"

    if write_header and save_to_csv:
        with open('experiment_h6.csv', 'a') as f:
            f.write('Model, Instance, Solve Time, objVal, Status, Best Algo' + '\n')

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
        bayes = Bayesian(delta_t=delta_t, T=T, time_horizon=time_horizon)
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
        nn = NN_Agent(delta_t=delta_t, T=T, time_horizon=time_horizon, max_iter=500)
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

if __name__ == "__main__":
    #xperiment(method='pcost_min', write_header=True, t=30, delta_t=5, T=300, time_horizon=6)
    #experiment(method='bayesian', t=30, delta_t=5, T=300, time_horizon=6)
    #experiment(method='nn', t=30, delta_t=5, T=300, time_horizon=6)
    experiment(method='rnn', t=30, delta_t=5, T=300, time_horizon=6)