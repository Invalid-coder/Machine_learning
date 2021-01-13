import numpy as np

def Predict(ex):
    s = w.dot(ex)
    print("Example: ", ex)
    print("Sum = ", s)

    if s > 0:
        return 1
    else:
        return 0

def Target(ex):
    if ex[1] == 1 and ex[2] == 0.3:
        return 1
    elif ex[1] == 0.4 and ex[2] == 0.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    w = np.array([[0, 0, 0]])
    examples = np.array([[1, 1, 0.3], [1, 0.4, 0.5], [1, 0.7, 0.8]])
    perfect = False

    while not perfect:
        perfect = True

        for ex in examples:
            print("Example: ", ex)
            if Predict(ex) != Target(ex):
                perfect = False
                if Predict(ex) == 0:
                    w = w + ex
                    print("w: ", w)
                else:
                    w = w - ex
                    print("w: ", w)

    print("Final result: ", w)
