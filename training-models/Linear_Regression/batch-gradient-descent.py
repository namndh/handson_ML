import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

initial_X = 2 * np.random.rand(400, 1)
y = 3 + 1.5 * initial_X + np.random.rand(400, 1)

X = np.c_[np.ones((400, 1)), initial_X]
X = np.matrix(X)
y = np.matrix(y)


def plot_gradient_descent(initial_X, X, y, theta, eta,):
    n_iteration = 1000
    m = len(X)
    plt.plot(initial_X, y, 'b.')
    for i in range(n_iteration):
        gradients = 2/m * X.T * (X*theta - y)
        theta = theta - eta*gradients
    print(theta)
    initial_X_new = np.linspace(0, 2, 10).reshape(10, 1)
    X_new = np.c_[np.ones((10, 1)), initial_X_new]
    y_new = X_new * theta

    plt.plot(initial_X_new, y_new, 'r-', linewidth=1, label='Predictions')
    plt.xlabel('X[1]')
    plt.ylabel('y')
    plt.axis([0, 2, 0, 7])
    plt.legend(loc='upper left')
    plt.title(r'$\eta = {}$'.format(eta))


np.random.seed(42)
initial_theta = np.random.randn(2, 1)

plt.figure(figsize=(20, 8))
plt.subplot(131)
plot_gradient_descent(initial_X, X, y, initial_theta, eta=0.005)
plt.subplot(132)
plot_gradient_descent(initial_X, X, y, initial_theta, eta=0.1)
plt.subplot(133)
plot_gradient_descent(initial_X, X, y, initial_theta, eta=0.5)

plt.show()