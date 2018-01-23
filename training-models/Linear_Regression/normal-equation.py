import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

initial_X = 3 * np.random.rand(100, 1)
y = 5 + 2 * initial_X + np.random.rand(100, 1)
print(initial_X.shape)
print(y.shape)
print(pd.DataFrame(initial_X).max())
print(pd.DataFrame(initial_X).min())
print(pd.DataFrame(y).max())
print(pd.DataFrame(y).min())
plt.plot(initial_X,y,'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0,3,5,12])
plt.show()
X = np.c_[np.ones((100, 1)), initial_X]
X = np.matrix(X)
y = np.matrix(y)
theta = np.linalg.inv(X.T * X)*X.T*y
print(theta)
initial_X_new = np.linspace(0, 3, 10).reshape(10, 1)
X_new = np.c_[np.ones((10,1)), initial_X_new]
y_new = X_new * theta
print(y_new)
plt.plot(initial_X,y,'b.')
plt.plot(initial_X_new,y_new,'r-',linewidth=2,label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0,3,5,12])
plt.legend(loc='upper left')
plt.show()