import numpy as np
from numpy import linalg

x = np.array([1, 60, 1, 50, 1, 75])
x.resize(3, 2)
y = np.array([10, 7, 12])
y.resize(3, 1)
res = linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print(res)