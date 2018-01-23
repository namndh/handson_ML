from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
poly_feature = PolynomialFeatures(degree=2, include_bias=False)


m = 100
X = 6 * np.random.rand(m,1) - 3
X = np.matrix(X)
y = 0.5*np.power(X,2) + X + 2 + np.random.rand(m,1)
print(type(X))
print(pd.DataFrame(X).min())
print(pd.DataFrame(y).min())
plt.plot(X, y, 'b.')
plt.axis([-3, 3, 0, 11])
plt.show()
X_poly = poly_feature.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_feature.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, "r-", linewidth=2, label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.axis([-3, 3, 0, 11])
plt.show()