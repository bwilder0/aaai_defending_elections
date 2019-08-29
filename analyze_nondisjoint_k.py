import pickle
import numpy as np
import matplotlib.pyplot as plt

vals_a, vals_d = pickle.load(open('results_yahoo_100.pickle', 'rb'))


k_a_vals = [5, 10, 20]
k_d_vals = [1, 3, 5, 10, 15, 20, 30, 40, 50]
k_a = 10
plt.plot([np.mean(vals_d[(k_a, k_d)]) for k_d in k_d_vals if (k_a, k_d) in vals_d])
plt.plot([np.mean(vals_a[(k_a, k_d)]) for k_d in k_d_vals if (k_a, k_d) in vals_d])

k_a = 5
plt.plot([np.mean(vals_d[(k_a, k_d)]) for k_d in k_d_vals if (k_a, k_d) in vals_d])
plt.plot([np.mean(vals_a[(k_a, k_d)]) for k_d in k_d_vals if (k_a, k_d) in vals_d])


#plt.ylim(0, 25)