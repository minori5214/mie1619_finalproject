"""
Solve all the instances in 30 seconds and record the incumbent best score. 

"""

import os
from mip_lazy import MIP

# Too large instance size, or unknown data type
skip_list = ["brd14051.tsp", "d15112.tsp", "d18512.tsp", 
            "dantzig42.tsp", "gr120.tsp", "pa561.tsp", 
            "rl11849.tsp", "usa13509.tsp", "bayg29.tsp", 
            "pla33810.tsp", "pla7397.tsp", "pla85900.tsp",
            "rl5915.tsp", "rl5934.tsp"]

too_large = ["fl3795.tsp", "fnl4461.tsp", "pcb3038.tsp"] # size > 3000 is omitted for the first report

def solve_all(dir, time_limit=300, save_to_csv=True):
    if save_to_csv:
        with open('output_{}.csv'.format('MIP'), 'a') as f:
            f.write('Model, Instance, NumConstr, Status, Opt_Cost, Solve_time, Cur_best' + '\n')

    for file in os.listdir(dir):
        if file.split('.')[-1] == 'tsp':
            print("file name: ", file)
            if file in skip_list or file in too_large:
                print("skipped")
                continue
            mip = MIP(os.path.join(dir, file), verbose=0)

            mip.build()
            result = mip.solve(time_limit=time_limit)

            NumConstrs = result.statistics['NumConstrs']
            solve_time = result.statistics['solve_time']
            cur_best = result.objVal
            status = result.statistics['status']
            if status == 9:
                optimal_cost = -1
            else:
                optimal_cost = cur_best

            if save_to_csv:
                with open('output_{}.csv'.format('MIP'), 'a') as f:
                    f.write('MIP' + ',' + file.split('.')[0] + ',' + str(NumConstrs) + ',' + str(status) + ','
                            + str(optimal_cost) + ',' + str(solve_time) + ',' + str(cur_best) + '\n'
                    )

if __name__ == "__main__":
    solve_all("instances", time_limit=300)
