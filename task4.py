import numpy as np
import sys

class Perceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward_pass(self, single_input):
        result = 0

        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]

        result += self.b

        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):
        return input_matrix.dot(self.w) > -self.b

    def train_on_single_example(self, example, y):
        predict = self.vectorized_forward_pass(example.T)[0][0]
        loss = y - predict
        self.w += example * loss
        self.b += loss

        return abs(loss)

    def MSE(self, y_hat, y):
        """
        Minimum square error function
        """
        return np.mean((y_hat - y) ** 2)

    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        """
        :param input_matrix: входные данные
        :param y: вектор правильных ответов
        :param max_steps: кол-во эпох
        :return:
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer[0])
                errors += int(error)  # int(True) = 1, int(False) = 0, так что можно не делать if

            train_loss = self.MSE(self.vectorized_forward_pass(input_matrix), y)
            sys.stdout.write("\rProgress: {}, Training loss: {}".format(str(100*i/float(max_steps))[:4],
                                                                        str(train_loss)[:5]))
if __name__ == '__main__':
    """
    Solving XOR with Neural Net
    """
    w = np.array([[0,0]]).T
    p1 = Perceptron(w.copy(), 0) # a1 = AND(not(a),b)
    p2 = Perceptron(w.copy(), 0) # a2 = AND(a, not(b))
    p3 = Perceptron(w.copy(), 0) # OR(a1, a2)
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y1 = np.array([[0, 0, 1, 0]]).T
    y2 = np.array([[0, 1, 0, 0]]).T
    y3 = np.array([[0, 1, 1, 1]]).T
    p1.train_until_convergence(X, y1)
    p2.train_until_convergence(X, y2)
    p3.train_until_convergence(X, y3)
    print(p1.w, p1.b)
    print(p2.w, p2.b)
    print(p3.w, p3.b)

    for example in X:
        print(example)
        a = p1.forward_pass(example)
        b = p2.forward_pass(example)
        print(p3.forward_pass([a, b]))

