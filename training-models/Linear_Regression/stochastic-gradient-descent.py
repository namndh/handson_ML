import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from learning_curves import plot_training_curves
nb_samples = 500

X, y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

print(X.shape)
print(y.shape)

print(pd.DataFrame(y).describe())
print(pd.DataFrame(X).describe())


def plot_data(X, y):
    fig, ax = plt.subplots(1, 1, figsize=(50, 30))

    ax.grid()
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')

    m = X.shape[0]

    for i in range(m):
        if y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


# plot_data(X, y)
m = X.shape[0]

X_b = np.c_[np.ones((nb_samples, 1)), X]
X_b = np.matrix(X_b)
# y = y.reshape(500,1)
n = X_b.shape[1]
# iteration = 10
# # learning schedule hyperparameter
# t0, t1 = 5, 50
#
#
# # def learning_schedule(t):
# #     return t0/(t + t1)
# #
# #
# # np.random.seed(1354654)
# # theta = np.random.randn(3, 1)
# #
# #
# # for i in range(iteration):
# #     for j in range(nb_samples):
# #         random_index = np.random.randint(nb_samples)
# #         x_j = X_b[random_index:random_index+1, :]
# #         y_j = y[random_index:random_index+1, :]
# #         gradients = 2 * x_j.T * (x_j*theta - y_j)
# #         eta = learning_schedule(i*m + j)
# #         theta = theta - eta*gradients
# #
# # print(theta)
# #
# # y_prediction2 = X_b*theta
# # print(pd.DataFrame(y_prediction2).describe())
# # predict2 = 0
# # for i in range(m):
# #     if y[i] == y_prediction2[i]:
# #         predict2 = predict2 + 1
# #
# #
# # print('Correctness of the manual optimization, iter = 40 :{}'.format(predict2/m * 100))


sgd = SGDClassifier(loss='perceptron', learning_rate='optimal', max_iter=40)

X_train, X_val, y_train, y_val = train_test_split(X_b, y, test_size=0.2, random_state=10)
train_errors, val_errors = [], []

for m in range(10, len(X_train)):
    sgd.fit(X_train[:m], y_train[:m])
    y_train_predict = sgd.predict(X_train[:m])
    y_val_predict = sgd.predict(X_val)
    train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
    val_errors.append(mean_squared_error(y_val_predict, y_val))

plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
plt.plot(np.sqrt(val_errors), 'b-', linewidth=2, label='validation')
plt.legend(loc='upper right')
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.axis([0, 100, -3, 3])
plt.show()


sgd.fit(X_b, y)
y_prediction = sgd.predict(X_b)
predict = 0
for i in range(m):
    if y[i] == y_prediction[i]:
        predict = predict + 1

print('Correctness of the built-in SGD classifier:{}%'.format(predict/m * 100))
