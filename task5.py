import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NN:
    def __init__(self, activation_function=sigmoid, activation_function_prime=sigmoid_prime):
        self.w1 = np.array([0.7,0.8,0.2,0.3,0.7,0.6]).reshape(2, 3)
        self.w2 = np.array([0.2,0.4]).reshape(1, 2)
        self.z = []
        self.a = []
        self.deltas = []
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime

    def forward_prop(self, X):
        z1 = self.w1.dot(X)
        a1 = np.array([z1[0] if z1[0][0] > 0 else 0, self.activation_function(z1[1])])
        z2 = self.w2.dot(a1)
        a2 = self.activation_function(z2)
        self.a.append(X)
        self.z.append(z1)
        self.a.append(a1)
        self.z.append(z2)
        self.a.append(a2)

        return z2

    def backward_prop(self, pred, ans):
        delta2 = (pred - ans) * self.activation_function_prime(self.z[1])
        delta1 = self.w2.T.dot(delta2) * self.activation_function_prime(self.z[0])
        self.deltas.append(delta1)
        self.deltas.append(delta2)

    def w_der(self, l, j, k):
        return self.a[l - 2][k - 1] * self.deltas[l - 2][j - 1]

if __name__ == '__main__':
    nn = NN()
    X = np.array([0,1.0,1.0]).reshape(3,1)
    nn.forward_prop(X)
    nn.backward_prop(nn.a[-1], 1)
    print(nn.w_der(2,1,3))
    print(nn.w_der(2,2,3))