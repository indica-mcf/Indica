import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def ordinary_least_squares():
    x = 2 * np.random.rand(100, 1)
    y = 4 + 5 * x + np.random.randn(100, 1)

    X = sm.add_constant(x)
    # print(type(X))

    model = sm.OLS(y, X)
    # print(dir(model))

    results = model.fit()
    # print(dir(results))

    # print(results.summary())
    # print('Parameters: ', results.params)
    # print('R2: ', results.rsquared)
    # print('Residuals: ', results.resid)

    y_fit = []
    for i in range(0, len(x)):
        y_fit.append(results.params[1] * x[i] + results.params[0])

    plt.plot(x, y, 'x')
    plt.plot(x, y_fit)
    plt.show()

    return results.params, results.rsquared

def linear_regr_scikit():
    # Load the diabetes dataset
    x = 2 * np.random.rand(100, 1)
    y = 4 + 5 * x + np.random.randn(100, 1)
    # Use only one feature
    # x = x[:, np.newaxis,0]

    # Split the data into training/testing sets
    x_train = x[:-80]
    x_test = x[-20:]

    # Split the targets into training/testing sets
    y_train = y[:-80]
    y_test = y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    params = regr.get_params()
    print(params)
    # The coefficients
    print("Coefficients: %.5f" % regr.coef_)
    print("y-intercept = %.5f" % regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.plot(x_test, y_test, 'x',color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=1)

    plt.show()



