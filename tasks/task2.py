import urllib
from urllib import request
import numpy as np

fname = input()  # read file name from stdin
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
x = np.copy(data)
x[:, 0] = 1
y = data[:, 0]
res = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print(*res)