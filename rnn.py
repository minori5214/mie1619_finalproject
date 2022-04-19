from time import time
import torch
import torch.nn as nn
import torch.optim as optim

from mip_lazy import MIP
from cp_v2 import CP
from adaptive_lns_v3 import ALNS_Agent
from result import Result

import sklearn.model_selection
import numpy as np
import os

ALGO_NUM = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size).to(device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class RNN_ReLU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN_ReLU, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU(output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size).to(device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden



class RNN_Switching():
    def __init__(self, algo_names=['MIP', 'CP', 'ALNS'], t=60, time_horizon=3, T=300, 
                    hidden_dim=5, n_layers=3, alpha=0.5,
                    lr=0.001, batch_size=4, max_iter=10000, X_test=[], y_test=[], model_name='rnn_switching'):

        self.algo_names = algo_names
        self.t = t
        self.time_horizon = time_horizon
        self.T = T
        self.alpha = alpha

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.model = RNN_ReLU(len(algo_names), len(algo_names),
                            hidden_dim,
                            n_layers).to(device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr)
        self.criterion = nn.MSELoss()

        self.X_test = X_test
        self.y_test = y_test

        if not os.path.exists('./models'):
            os.mkdir('./models')
        self.model_path = os.path.join('./models', model_name) + '.pth'

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Successfully loaded the model!")
        else:
            print("Model was not found. Please train the model from scratch")

    def fit(self, X, y, load=True):
        assert X.shape[1] == len(self.algo_names) * self.time_horizon, \
            "Train data size does not match with the model size"

        if load and os.path.exists(self.model_path):
            self.load_model()
            return

        train_losses = []
        test_losses = []
        for e in range(self.max_iter):
            self.model.train()
            epoch_loss = 0

            for i in range(0, len(y), self.batch_size):
                X_batch = X[i:min(i+self.batch_size, len(y)-1), :].reshape(-1, self.time_horizon, len(self.algo_names))
                X_batch = torch.from_numpy(X_batch).float().to(device)
                y_batch = torch.from_numpy(y[i:min(i+self.batch_size, len(y)-1), :]).float().to(device)

                self.optimizer.zero_grad()
                output, hidden = self.model(X_batch)

                y_pred = output[2::self.time_horizon]

                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if len(self.X_test) != 0 and len(self.y_test) != 0:
                test_loss, test_acc = self.score(self.X_test, self.y_test)

            if e % 100 == 0:
                if len(self.X_test) != 0 and len(self.y_test) != 0:
                    print("Episode {}: train_loss: {:.3f}, test_loss: {:.3f}".format(e, epoch_loss, test_loss))
                    test_losses.append(test_loss.item())
                else:
                    print("Episode {}: train_loss: {:.3f}".format(e, epoch_loss))

            train_losses.append(epoch_loss)

            torch.save(self.model.state_dict(), self.model_path)
        
        return train_losses, test_losses

    def run(self, instance_path):
        self.instance_path = instance_path

        self.base = {}
        self.base_id = {}
        k = 0
        if 'MIP' in self.algo_names:
            self.base['MIP'] = MIP(instance_path, verbose=0)
            self.base_id[k] = 'MIP'
            self.base['MIP'].build()
            k += 1
        if 'CP' in self.algo_names:
            self.base['CP'] = CP(instance_path, verbose=0)
            self.base_id[k] = 'CP'
            self.base['CP'].build()
            k += 1
        if 'ALNS' in self.algo_names:
            self.base['ALNS'] = ALNS_Agent(instance_path, verbose=0)
            self.base_id[k] = 'ALNS'
            self.base['ALNS'].build()

        weight = [1/len(self.base)]*len(self.base) # weights are set equal
        best_result = Result(float('inf'), None, None)
        num_iter = self.T // self.t
        solve_time = 0
        input_to_rnn = [] # queue
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

            non_inf_max_prev = max([x for x in performance_prev if x != float('inf')])
            non_inf_max = max([x for x in performance if x != float('inf')])
            performance_prev = [x if x != float('inf') else non_inf_max_prev for x in performance_prev]
            performance = [x if x != float('inf') else non_inf_max for x in performance]

            cost_improvement = [(performance_prev[j]-performance[j])/(self.t*weight[j]) for j in range(len(self.base))]
            if len(input_to_rnn) >= len(self.base):
                input_to_rnn.pop(0)
            input_to_rnn.append(cost_improvement)

            if i == 0:
                try:
                    non_nan_maxima = max([x for x in performance if x != float('inf')])
                except:
                    non_nan_maxima = float('inf')
            elif i >= 3: # Weights do not change in the first three iterations
                pred_cost_improvement = self.predict(np.array(input_to_rnn).reshape(1, -1))[0].cpu().detach().numpy()

                # Normalize
                if sum(pred_cost_improvement) != 0:
                    coeff = 1/sum(pred_cost_improvement)
                    pred_cost_improvement = [x*coeff for x in pred_cost_improvement]
                else:
                    pred_cost_improvement = [1/3, 1/3, 1/3] # all equal

                weight = [(1-self.alpha)*weight[j]+self.alpha*pred_cost_improvement[j] for j in range(len(self.base))]
                assert abs(1.0 - sum(weight)) < 0.0001, \
                        "Invalid weight={}. performance_prev={}, \
                        performance={}, cost_improvement={}.".format(
                            weight, performance_prev, performance, cost_improvement)
                print(i, weight, best_result.objVal)

                performance_prev = performance
        
        return best_result.objVal, solve_time, result.statistics['status'], weight_history

    def predict(self, X):
        self.model.eval()

        print("pred", X.shape)
        X = torch.from_numpy(X).float().to(device).reshape(-1, self.time_horizon, len(self.algo_names))

        self.optimizer.zero_grad()
        output, hidden = self.model(X)
        y_pred = output[2::self.time_horizon]

        return y_pred

class RNN_Agent():
    def __init__(self, algo_names=['MIP', 'CP', 'ALNS'], delta_t=5, time_horizon=6, t=90, T=600, 
                    hidden_dim=5, n_layers=3, 
                    lr=0.001, batch_size=16, max_iter=500, X_test=[], y_test=[], model_name='rnn'):
        """
        It solves the instance for "delta_t * time_horizon" [sec] by each method.
        Performance metric at each delta_t [sec] is recorded and then used for 
        determining the algorithm to use for the rest "T-delta_t*time_horizon*num(algorithms)" [sec]. 
        
        """
        self.algo_names = algo_names
        self.delta_t = delta_t
        self.time_horizon = time_horizon
        self.T = T
        self.t = t

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.model = RNN(len(algo_names), len(algo_names),
                            hidden_dim,
                            n_layers).to(device)

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
        remain = self.T-self.t
        result = self.base[best_algo].resume(remain)

        objVal = result.objVal
        if best_algo in ['MIP', 'CP']:
            solve_time = result.statistics['solve_time']
            status = result.statistics['status']
        elif best_algo == 'ALNS':
            solve_time = remain
            status = 9

        return objVal, solve_time+self.t, status, best_algo

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Successfully loaded the model!")
        else:
            print("Model was not found. Please train the model from scratch")

    def fit(self, X, y, load=True):
        assert X.shape[1] == len(self.algo_names) * self.time_horizon, \
            "Train data size does not match with the model size. X.shape={}, expected={}".format(X.shape, len(self.algo_names) * self.time_horizon)

        if load and os.path.exists(self.model_path):
            self.load_model()
            return

        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for e in range(self.max_iter):
            self.model.train()
            epoch_loss = 0
            correct_num = 0

            for i in range(0, len(y), self.batch_size):
                X_batch = X[i:min(i+self.batch_size, len(y)-1), :].reshape(-1, self.time_horizon, len(self.algo_names))
                X_batch = torch.from_numpy(X_batch).float().to(device)
                y_batch = torch.from_numpy(y[i:min(i+self.batch_size, len(y)-1)]).long().to(device)

                self.optimizer.zero_grad()
                output, hidden = self.model(X_batch)

                y_pred_logsoftmax = torch.log_softmax(output[2::self.time_horizon], dim = 1)
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

            #torch.save(self.model.state_dict(), self.model_path)
        
        return train_losses, train_accs, test_losses, test_accs

    def predict(self, X):
        self.model.eval()

        X = torch.from_numpy(X).float().to(device).reshape(-1, self.time_horizon, len(self.algo_names))

        self.optimizer.zero_grad()
        output, hidden = self.model(X)
        y_pred_logsoftmax = torch.log_softmax(output[2::self.time_horizon], dim = 1)
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
    """
    delta_t = 5
    T = 300
    time_horizon = 3

    X = np.load('X_t{}_T{}_h3.npy'.format(delta_t, T))
    y = np.load('y_t{}_T{}_h3.npy'.format(delta_t, T))

    dataset = Dataset(X, y)
    X_train, X_test, y_train, y_test = dataset.train_test_split()
    rnn = RNN_Agent(delta_t=delta_t, T=T, time_horizon=time_horizon, X_test=X_test, y_test=y_test)
    rnn.fit(X_train, y_train, load=False)
    #objVal = rnn.run('./instances/gr48.tsp')
    #print(objVal)
    """

    t = 20
    T = 300
    h = 3

    X = np.load('X_rnn_t{}_T{}_h{}.npy'.format(t, T, h))
    y = np.load('y_rnn_t{}_T{}_h{}.npy'.format(t, T, h))
    dataset = Dataset(X, y)
    X_train, X_test, y_train, y_test = dataset.train_test_split()

    rnn_switching = RNN_Switching(t=t*ALGO_NUM)
    rnn_switching.fit(X_train, y_train)
    objVal = rnn_switching.run('./instances/att532.tsp')