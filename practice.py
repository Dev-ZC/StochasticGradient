import numpy as np

true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []

for i in range(10000):
    x = np.random.rand(d)
    y = true_w + np.random.rand()
    print(x, y)
    points.append(x, y)
    
# Squared Loss (squared residual)
def f(w, i):
    return 
