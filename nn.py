import torch
import torch.nn as nn
import torch.optim as optim

from mip_lazy import MIP
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent

import sklearn.model_selection
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCN(nn.Module):
    """
    Fully connected neural networks.

    """
    def __init__(self, input_size, output_size):
        super(FCN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(96, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class NN_Agent():
    def __init__(self, algo_names=['MIP', 'CP', 'ALNS'], delta_t=5, time_horizon=3, T=300, 
                    lr=0.001, batch_size=32, max_iter=200, X_test=[], y_test=[], model_name='nn'):
        """
        It solves the instance for "delta_t * time_horizon" [sec] by each method.
        Performance metric at each delta_t [sec] is recorded and then used for 
        determining the algorithm to use for the rest "T-delta_t*time_horizon*num(algorithms)" [sec]. 
        
        """
        self.algo_names = algo_names
        self.delta_t = delta_t
        self.time_horizon = time_horizon
        self.T = T

        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.model = FCN(len(algo_names)*time_horizon, len(algo_names)).to(device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.X_test = X_test
        self.y_test = y_test

        if not os.path.exists('./models'):
            os.mkdir('./models')
        self.model_path = os.path.join('./models', model_name) + '.pth'
    
    def run(self, instance_path):
        self.instance_path = instance_path

        self.base = {}
        self.base_id = {}
        k = 0
        if 'MIP' in self.algo_names:
            self.base['MIP'] = MIP(instance_path, verbose=0)
            self.base_id[k] = 'MIP'
            k += 1
        if 'CP' in self.algo_names:
            self.base['CP'] = CP(instance_path, verbose=0)
            self.base_id[k] = 'CP'
            k += 1
        if 'ALNS' in self.algo_names:
            self.base['ALNS'] = ALNS_Agent(instance_path, verbose=0)
            self.base_id[k] = 'ALNS'


        input_to_nn = []
        for i in range(self.time_horizon):
            performance = []

            # For each algorithm, run for delta_t seconds
            for algo in self.base:
                if i == 0:
                    # If first time, build and solve
                    self.base[algo].build()
                    result = self.base[algo].solve(time_limit=self.delta_t)
                else:
                    # From the second time, just resume optimization
                    result = self.base[algo].resume(time_limit=self.delta_t)

                obj = result.objVal
                performance.append(obj)
            
            metric = [1 if x == min(performance) else 0 for x in performance] # best=1, behind=0
            input_to_nn.extend(metric)
        
        # Choose the algorithm
        algo_id, _ = self.predict(np.array(input_to_nn).reshape(1, -1))
        algo_id = algo_id.item()
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
        
        self.model.train()

        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for e in range(self.max_iter):
            self.model.train()
            epoch_loss = 0
            correct_num = 0

            for i in range(0, len(y), self.batch_size):
                X_batch = torch.from_numpy(X[i:min(i+self.batch_size, len(y)-1), :]).float().to(device)
                y_batch = torch.from_numpy(y[i:min(i+self.batch_size, len(y)-1)]).long().to(device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)
                y_pred_logsoftmax = torch.log_softmax(output, dim = 1)
                _, y_pred = torch.max(y_pred_logsoftmax, dim = 1)
                
                loss = self.criterion(y_pred_logsoftmax, y_batch)
                loss.backward()
                self.optimizer.step()

                correct_num += (y_pred == y_batch).float().sum()
                epoch_loss += loss.item()

            train_acc = torch.round(correct_num / len(y) * 100)

            if len(self.X_test) != 0 and len(self.y_test) != 0:
                test_loss, test_acc = self.score(self.X_test, self.y_test)

            if e % 10 == 0:
                if len(self.X_test) != 0 and len(self.y_test) != 0:
                    print("Episode {}: train_loss: {:.3f}, train_acc: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}".format(e, epoch_loss, train_acc, test_loss, test_acc))
                    test_losses.append(test_loss.item())
                    test_accs.append(test_acc.item())
                else:
                    print("Episode {}: train_loss: {:.3f}, train_acc: {:.3f}".format(e, epoch_loss, train_acc))

            train_losses.append(epoch_loss)
            train_accs.append(train_acc.item())

            torch.save(self.model.state_dict(), self.model_path)
        
        return train_losses, train_accs, test_losses, test_accs

    def predict(self, X):
        self.model.eval()

        X = torch.from_numpy(X).float().to(device)

        self.optimizer.zero_grad()
        output = self.model(X)
        y_pred_logsoftmax = torch.log_softmax(output, dim = 1)
        _, y_pred = torch.max(y_pred_logsoftmax, dim = 1)

        return y_pred, y_pred_logsoftmax

    def score(self, X, y):
        y = torch.from_numpy(y).long().to(device)
        y_pred, y_pred_logsoftmax = self.predict(X)

        loss = self.criterion(y_pred_logsoftmax, y)
        correct_num = (y_pred == y).float().sum()
        test_acc = torch.round(correct_num / len(y) * 100)

        #print("Test: loss: {:.3f}, acc: {:.3f}".format(loss, test_acc))

        return loss, test_acc


class Dataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_test_split(self, train_size=None, test_size=None, shuffle=True):
        if test_size == None:
            if train_size != None:
                assert 0 <= train_size <= 1.0, \
                    "Invalid train size (={}). Set the range between 0.0 and 1.0".format(train_size)
                test_size = 1.0 - train_size
            else:
                test_size = 0.25
        if train_size == None:
            train_size = 1.0 - test_size
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(self.X, self.y, train_size=train_size, test_size=test_size, shuffle=shuffle)
        return X_train, X_test, y_train, y_test

    def __getitem__(self, index):
        return self.X[index, :], self.y[index, :]

    def __len__ (self):
        return len(self.y)


if __name__ == "__main__":
    delta_t = 5
    T = 300
    time_horizon = 3

    X = np.load('X_t{}_T{}.npy'.format(delta_t, T))
    y = np.load('y_t{}_T{}.npy'.format(delta_t, T))

    dataset = Dataset(X, y)
    X_train, X_test, y_train, y_test = dataset.train_test_split()
    nn = NN_Agent(X_test=X_test, y_test=y_test)

    nn.fit(X_train, y_train)
    #objVal = nn.run('./instances/gr48.tsp')
    #print(objVal)