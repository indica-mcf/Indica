"""
This is a test script for writing OLS, WLS (and more) models for linear regression analysis

"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def test_ordinary_least_squares():
    x = np.linspace(0, 10, 10)
    X = sm.add_constant(x)
    # print(type(X))

    y = [1, 3, 3, 4, 6, 5, 7, 8, 10, 9.5]

    model = sm.OLS(y, X)
    # print(dir(model))

    results = model.fit()
    # print(dir(results))

    # print(results.summary())
    print('Parameters: ', results.params)
    print('R2: ', results.rsquared)
    print('Residuals: ', results.resid)

    y_fit = []
    for i in range(0, len(x)):
        y_fit.append(results.params[1] * x[i] + results.params[0])

    plt.plot(x, y, 'x')
    plt.plot(x, y_fit)
    plt.show()

    return


# def weighted_least_squares():
#     nsample = 10
#     x = np.linspace(0, 10, nsample)
#     X = np.column_stack((x, x ** 2))
#
#     X = sm.add_constant(X)
#     w = np.ones(nsample)
#     w[nsample * 0 //10:] = 10
#     print(w)
#     y = [1, 3, 3, 4, 6, 5, 7, 8, 10, 9.5]
#
#     mod_wls = sm.WLS(y, X, weights=1 / w)
#     print('yes')
#     res_wls = mod_wls.fit()
#     print(res_wls.summary())


def ordinary_least_squares(x_variable: np.ndarray, y_variable: np.ndarray):
    """

    Parameters
    ----------
    x_variable
    y_variable

    Returns
    -------
    parameter fit (gradient and y-intercept)
    R-squared value (goodness of fit)
    """
    X = sm.add_constant(x_variable)

    regr_model = sm.OLS(y_variable, X)
    results = regr_model.fit()

    return results.params, results.rsquared


def cal_cost(theta, X, y):
    '''

    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))

    where:
        j is the no of features
    '''

    m = len(y)

    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)

        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)

    return theta, cost_history, theta_history


lr =0.01
n_iter = 1000

X = 2 * np.random.rand(100,1)
y = 4 +3 * X + np.random.randn(100,1)

print(X)
print(y)

theta = np.random.randn(2,1)
X_b = np.c_[np.ones((len(X),1)),X]


theta,cost_history,theta_history = gradient_descent(X_b,y,theta,lr,n_iter)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
print('Gradient = ' + str(theta[0][0]), 'y-intercept: = ' + str(theta[1][0]))
y_new = []
for i in range(0,len(X)):
    y_new.append(theta[0][0] * X[i] + theta[1][0])

plt.plot(X, y, 'x')
plt.plot(X, y_new)
plt.show()