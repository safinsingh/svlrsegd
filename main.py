import numpy as np

# yes this is an antipattern
x_train = np.array([1, 2, 3, 4])
y_train = np.array([2, 5, 6, 8.5])

assert x_train.shape[0] == y_train.shape[0]
m = x_train.shape[0]


# f(x) = wx + b
# sse = 1/2m * sum(i=1 to m)[(f(x(i)) - y(i))^2]
def sum_squared_error(x_train, y_train, w, b):
    ret = 0
    for i in range(m):
        pred = w * x_train[i] + b
        error = pow(pred - y_train[i], 2)
        ret += error
    return ret


# w = w_prev - alpha * dJ/dw
# b = b_prev - alpha * dJ/db
# dJ/db = 1/m * sum(i=1 to m)[(f(x(i)) - y(i))]
# dJ/dw = 1/m * sum(i=1 to m)[(f(x(i)) - y(i))]x(i)
def gradient_descent(x_train, y_train, w, b, alpha):
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        pred = w * x_train[i] + b
        diff = pred - y_train[i]
        dj_dw += diff * x_train[i]
        dj_db += diff

    return w - alpha * dj_dw, b - alpha * dj_db


def main():
    steps = 10000
    alpha = 1e-2
    # meow
    epsilon = 1e-6

    w, b = gradient_descent(x_train, y_train, 0, 0, alpha)
    sse = sum_squared_error(x_train, y_train, w, b)
    sse_prev = sse

    for i in range(steps):
        w, b = gradient_descent(x_train, y_train, w, b, alpha)
        sse = sum_squared_error(x_train, y_train, w, b)
        sse_diff = abs(sse - sse_prev)

        print(f"step {i}:   y={w}x+{b} | sse={sse} | sse_diff={sse_diff}")
        if sse_diff < epsilon:
            break

        sse_prev = sse

    print(f"optimal linear regression line: y={w}x+{b}")


if __name__ == "__main__":
    main()
