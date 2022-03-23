import os

from mip_lazy import MIP
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent

import csv
import numpy as np
from copy import copy

def make_raw_data(dir, delta_t=5, T=300, save_to_csv=True, model_name='MIP'):
    """
    Solve all the instances in 'dir' with the model ('model_name'), and
    record the performance at every 'delta_t' [sec] until it reaches optimality
    or 'T' [sec] passes.

    save_to_csv (bool): Save the record to csv if True.
    
    """
    if save_to_csv:
        with open('rawdata_{}_t{}_T{}.csv'.format(model_name, delta_t, T), 'a') as f:
            f.write('Model, Instance, t, objVal, Status' + '\n')

    for file in os.listdir(dir):
        print("file name: ", file)
        t = 0
        status = 9

        # Initialize the model
        if model_name == 'MIP':
            model = MIP(os.path.join(dir, file), verbose=0)
        if model_name == 'CP':
            model = CP(os.path.join(dir, file), verbose=0)
        if model_name == 'ALNS':
            model = ALNS_Agent(os.path.join(dir, file), verbose=0)

        while status != 2 and t < T:
            if t == 0:
                # Build and solve
                model.build()
                result = model.solve(time_limit=delta_t)
            else:
                # Resume optimization
                result = model.resume(time_limit=delta_t)

            # Retrieve the result
            objVal = result.objVal
            if model_name in ['MIP', 'CP']:
                solve_time = result.statistics['solve_time']
                status = result.statistics['status']
            elif model_name == 'ALNS':
                solve_time = delta_t
                status = 9

            t += delta_t if status != 2 else solve_time

            if save_to_csv:
                with open('rawdata_{}_t{}_T{}.csv'.format(model_name, delta_t, T), 'a') as f:
                    f.write(model_name + ',' + file.split('.')[0] + ',' + str(t) + ','
                            + str(objVal) + ',' + str(status) + '\n'
                    )

def make_dataset(delta_t, T, time_horizon=3):
    num_algos = 3 # MIP, CP, ALNS

    csv_mip = open('rawdata_MIP_t{}_T{}.csv'.format(delta_t, T), "r")
    f = csv.reader(csv_mip, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    next(f)
    mip_result = []
    for row in f:
        # row[1]: instance, [2]: time, [3]: objVal, [4]: status
        mip_result.append((row[1], 
                        float(row[2]), 
                        float(row[3]), 
                        int(row[4]))
                    )

    csv_cp = open('rawdata_CP_t{}_T{}.csv'.format(delta_t, T), "r")
    f = csv.reader(csv_cp, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    next(f)
    cp_result = []
    for row in f:
        cp_result.append((row[1], 
                        float(row[2]), 
                        float(row[3]), 
                        int(row[4]))
                    )

    csv_alns = open('rawdata_ALNS_t{}_T{}.csv'.format(delta_t, T), "r")
    f = csv.reader(csv_alns, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    next(f)
    alns_result = []
    for row in f:
        alns_result.append((row[1], 
                        float(row[2]), 
                        float(row[3]), 
                        int(row[4]))
                    )

    raw_performance = {}
    ys = []

    X_raw = []
    m_2, c_2, a_2 = float('inf'), float('inf'), float('inf')

    m = mip_result.pop(0)
    c = cp_result.pop(0)
    a = alns_result.pop(0)
    while len(mip_result) > 0 or len(cp_result) > 0 or len(alns_result) > 0:
        X_raw.append((m[2], c[2], a[2]))

        # when all models reach 9 or T [sec] passed, move on to the next instance
        if (m[3] == 2 and c[3] == 2 and a[3] == 2) or (max(m[1], c[1], a[1]) > T-delta_t):
            raw_performance[m[0]] = X_raw

            #print("check: ", m[0], X_raw[-1], np.argmin(X_raw[-1]))
            y = np.argmin(X_raw[-1])
            if X_raw[-1].count(y) > 1: # multiple algorithms reached optimality
                y = np.argmin((m_2, c_2, a_2))
            ys.append(y)

            # Initialize temp variabls
            X_raw = []
            m_2, c_2, a_2 = float('inf'), float('inf'), float('inf')

            m = mip_result.pop(0)
            c = cp_result.pop(0)
            a = alns_result.pop(0)
        else:
            # if previous status is not 2, read the next line
            # if 2, keep the previous objVal
            # m_2, c_2, a_2 = time when it reached optimality
            if m[3] !=2: m = mip_result.pop(0)
            else: m_2 = min(m_2, m[1])

            if c[3] !=2: c = cp_result.pop(0)
            else: c_2 = min(c_2, c[1])

            if a[3] !=2: a = alns_result.pop(0)
            else: c_2 = min(c_2, c[1])
    
    raw_performance[m[0]] = X_raw

    #print("check: ", m[0], X_raw[-1], np.argmin(X_raw[-1]))
    y = np.argmin(X_raw[-1])
    if X_raw[-1].count(y) > 1: # multiple algorithms reached optimality
        y = np.argmin((m_2, c_2, a_2))
    ys.append(y)

    #for instance, y in zip(raw_performance, ys): 
    #    print(raw_performance[instance], y)

    # convert raw performance to performance metric 
    # (e.g., (3320, 3500, 3300) -> (behind, behind, best))
    performance = {}
    for instance in raw_performance:
        X = []
        for x_raw in raw_performance[instance]:
            X.append(tuple([int(x == min(x_raw)) for x in x_raw]))
        performance[instance] = X
    
    X_train = []
    y_train = []
    for i, instance in enumerate(performance):
        print(i, instance, ys[i])
        x_train = []
        for x in performance[instance]:
            if len(x_train) < time_horizon * num_algos:
                x_train.extend(x)
            else:
                X_train.append(copy(x_train))
                y_train.append(ys[i])

                # pop old data
                for j in range(num_algos):
                    x_train.pop(0)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(y_train)

    #np.save('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon), X_train)
    #np.save('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon), y_train)

    #np.load('X_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))
    #np.load('y_t{}_T{}_h{}.npy'.format(delta_t, T, time_horizon))

    print("Dataset complete", X_train.shape, y_train.shape)

if __name__ == "__main__":
    #make_raw_data("instances_solvable_2", delta_t=5, T=300, model_name='MIP')
    #make_raw_data("instances_solvable_2", delta_t=5, T=300, model_name='CP')
    #make_raw_data("instances_solvable_2", delta_t=5, T=300, model_name='ALNS')
    make_dataset(delta_t=5, T=300, time_horizon=3)