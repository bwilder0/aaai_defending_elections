import pickle
import numpy as np
from nondisjoint import oga, attacker_br_value, attacker_br_mip_value, defender_br_mip_value

def load_instance(n, i, num_targets):
    with open('budget_instances/yahoo_' + str(n) + '_' + str(i), 'rb') as f:
        Pfull, wfull = pickle.load(f, encoding='bytes')
    P = np.zeros((n, num_targets), dtype=np.float32)
    for i in range(num_targets):
        for j in Pfull[i]:
            P[j, i] = Pfull[i][j]
    return P


k_a_vals = [5, 10, 20]
k_d_vals = [1, 3, 5, 10, 15, 20, 30, 40, 50]
num_iters = 30

num_channels = 100
num_voters = 500
num_iter = 50

vals_a = {}
vals_d = {}

for k_a in k_a_vals:    
    for k_d in k_d_vals:
        vals_a[(k_a, k_d)] = np.zeros(num_iters)
        vals_d[(k_a, k_d)] = np.zeros(num_iters)
        for i in range(num_iters):
            P = load_instance(num_channels, i, num_voters)
            preferences = np.random.binomial(1, 0.5, num_voters)
            sigma_d, x_a, values = oga(P, preferences, k_d, k_a, num_iter, step_size = 0.01, verbose=False)
            vals_a[(k_a, k_d)][i] = attacker_br_mip_value(sigma_d, P, preferences, k_a)
            vals_d[(k_a, k_d)][i] = defender_br_mip_value(x_a, P, preferences, k_d)
            print(k_a, k_d, i, vals_a[(k_a, k_d)][i], vals_d[(k_a, k_d)][i])
        pickle.dump((vals_a, vals_d), open('results_expo_yahoo_100.pickle', 'wb'))