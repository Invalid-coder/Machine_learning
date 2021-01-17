import numpy as np

def J_quadratic(y, y_hat):
    return 0.5 * np.mean((y_hat - y) ** 2)

def cross_entropy_loss(y, y_hat):
    print((np.log(1 - 0.3) + np.log(1 - 0.1)) / 3)
    #return np.mean(y*np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

if __name__ == '__main__':
    y = np.array([1, 0, 0]).reshape(3,1)
    y_hat = np.array([1, 0.3, 0.1]).reshape(3,1)
    print(J_quadratic(y, y_hat))
    print(cross_entropy_loss(y, y_hat))
