import numpy as np

n_turbs = 3
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
b = np.zeros_like(a)
freq = np.array([10, 20, 30])

for i, f in enumerate(freq):
    b[i * n_turbs: i * n_turbs + n_turbs] = a[i * n_turbs: i * n_turbs + n_turbs] * f

print(b)