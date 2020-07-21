"""
    This program implements logistic regression with regularization.
    By: Rahul Golhar
"""
import numpy
from ipython_genutils.py3compat import xrange
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy import optimize


# It takes the cost function and minimizes with the "downhill simplex algorithm."
# http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.fmin.html
# Thanks to David Kaleko for this: https://github.com/kaleko/CourseraML/blob/master/ex2/ex2.ipynb
def optimizeTheta(theta, X, y, regTermLambda=0.):
    """
        This function optimizes the cost function value to
        give theta and minimum cost value.
    :param theta:           theta to be used
    :param X:               X matrix
    :param y:               y vector
    :param regTermLambda:   regularization term(= 0 by default)
    :return:                optimal theta values and minimum cost value
    """
    print("\t==================> Optimizing cost function <==================\n")
    result = optimize.minimize(costFunction, theta, args=(X, y, regTermLambda), method='BFGS',
                               options={"maxiter": 500, "disp": False})

    return numpy.array([result.x]), result.fun


def h(theta, X):
    """
        This function returns the hypothesis value. (x*theta)
    :param theta:   the theta vector
    :param X:       the X matrix
    :return:        hypothesis value vector
    """
    return expit(numpy.dot(X,theta))


def featureMapping(x1col, x2col):
    """
        This function creates more features from each data point.
        Basically, we map features into polynomial terms of x1 and
        x2 till 6th power. This allows to built more accurate models
        but it makes it prone to overfitting.

        This code was taken from someone else.
    :param x1col:       feature x1
    :param x2col:       feature x2
    :return:            new feature
    """

    maxDegrees = 8
    result = numpy.ones((x1col.shape[0], 1))
    i=1
    while i <= maxDegrees:
        j = 0
        while j <= i:
            x1Power = x1col ** (i - j)
            x2Power = x2col ** (j)
            finalFeature = (x1Power * x2Power).reshape(x1Power.shape[0], 1)
            result = numpy.hstack((result, finalFeature))
            j += 1
        i+=1
    return result


def costFunction(theta, X, y, regTermLambda = 0.0):
    """
        This function returns the values calculated by the
        cost function.
    :param theta:   theta vector to be used
    :param X:               X matrix
    :param y:               y Vector
    :param regTermLambda:   regularization factor
    :return:                Cost Function results
    """
    m = y.size
    y_log_hx = numpy.dot(-numpy.array(y).T, numpy.log(h(theta, X)))
    one_y_log_one_hx = numpy.dot((1 - numpy.array(y)).T, numpy.log(1 - h(theta, X)))

    regTerm = (regTermLambda / 2) * numpy.sum(numpy.dot(theta[1:].T, theta[1:]))

    return float((1. / m) * (numpy.sum(y_log_hx - one_y_log_one_hx) + regTerm))


def plotInitialData(positiveExamples, negativeExamples):
    """
        This function plots the initial data and saves it.
    :param positiveExamples:    positive examples
    :param negativeExamples:    negative examples
    :return:    None
    """
    print("\n\tPlotting the initial data.")
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(positiveExamples[:, 1], positiveExamples[:, 2], 'k+', label='y=1')
    plt.plot(negativeExamples[:, 1], negativeExamples[:, 2], 'yo', label='y=0')
    plt.title('Initial data: Results of Microchip tests')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    plt.legend()
    plt.savefig("initialDataPlot.jpg")
    print("\tSaved the initial data plotted to initialDataPlot.jpg.")


def plotDecisionBoundary(theta, X, y, lambdaVal=0.0):
    """
    This function plots decision boundary for given theta, X, Y
    and lambda value. It first maps the features and minimizes
    the cost function using all parameters passed. It creates
    grids as x1 and x2. And for them, compute if the hypothesis
    classifies that point as Positive or Negative. And in the end,
    a contour is drawn.

    :param theta:       the theta value passed
    :param X:           X matrix
    :param y:           y Vector
    :param lambdaVal:   reg parameter value
    :return:            None
    """

    theta, minimumCost = optimizeTheta(theta, X, y, lambdaVal)
    x1 = numpy.linspace(-1, 1.5, 50)
    x2 = numpy.linspace(-1,1.5,50)
    hypoValues = numpy.zeros((len(x1), len(x2)))
    i=0
    while i < len(x1):
        j = 0
        while j < len(x2):
            mappingOf_i_j = featureMapping(numpy.array([x1[i]]), numpy.array([x2[j]]))
            hypoValues[i][j] = numpy.dot(theta,mappingOf_i_j.T)
            j += 1
        i+=1

    hypoValues = hypoValues.transpose()

    u, v = numpy.meshgrid(x1, x2)

    contourToPlot = plt.contour(x1, x2, hypoValues, [0])

    #Kind of a hacky way to display a text on top of the decision boundary
    # myfmt = {0:'Lambda = %d' % lambdaVal}

    plt.clabel(contourToPlot)

    plt.title("Decision Boundary")


def plotData(positiveExamples,negativeExamples):
    """
        This function plots the given data and saves it.
    :param positiveExamples:    positive examples
    :param negativeExamples:    negative examples
    :return:    None
    """
    plt.plot(positiveExamples[:,1],positiveExamples[:,2],'k+',label='y=1')
    plt.plot(negativeExamples[:,1],negativeExamples[:,2],'yo',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)


def plotDifferentBoundaries(theta, mappedFeatures, y, positiveExamples, negativeExamples):
    """
        This function plots the different decision boundaries for data and save them.

    :param theta:               theta vector
    :param mappedFeatures:      mapped X Matrix
    :param y:                   y vector
    :param positiveExamples:    positive examples
    :param negativeExamples:    negative examples
    :return:                    None
    """

    print("\n\tPlotting the decision bundaries.")

    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plotData(positiveExamples, negativeExamples)
    plotDecisionBoundary(theta, mappedFeatures, y, 0.0)

    plt.subplot(222)
    plotData(positiveExamples, negativeExamples)
    plotDecisionBoundary(theta, mappedFeatures, y, 1.0)

    plt.subplot(223)
    plotData(positiveExamples, negativeExamples)
    plotDecisionBoundary(theta, mappedFeatures, y, 10.0)

    plt.subplot(224)
    plotData(positiveExamples, negativeExamples)
    plotDecisionBoundary(theta, mappedFeatures, y, 100.0)

    plt.savefig("decisionBoundaries.jpg")
    print("\tSaved the decision boundaries plotted to decisionBoundaries.jpg.")


def main():
    """
        This is the main function.
    :return: None
    """
    print("******************* Starting execution **********************")

    # Read the data
    data = numpy.loadtxt('data/microchip.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)

    print("\nSuccessfully read the data.")

    # ***************************************** Step 1: Initial data *****************************************
    print("Getting the data ready.")

    # X matrix
    X = numpy.transpose(numpy.array(data[:-1]))
    # y vector
    y = numpy.transpose(numpy.array(data[-1:]))
    # no of training examples
    m = y.size
    # Insert a column of 1's into the X matrix
    X = numpy.insert(X, 0, 1, axis=1)

    # Divide the sample into two: ones with positive classification, one with negative classification
    positiveExamples = numpy.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
    negativeExamples = numpy.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])

    # plot initial data on screen
    plotInitialData(positiveExamples,negativeExamples)


    # ***************************************** Step 2: Feature Mapping *****************************************
    # Create more feature from current features
    mappedFeatures = featureMapping(X[:, 1], X[:, 2])

    # ***************************************** Step 3: Optimize the cost function *****************************************

    # For theta = zeros the cost function returns the value around 0.693
    initial_theta = numpy.zeros((mappedFeatures.shape[1], 1))

    print("\n\tResult of cost function with theta = ",costFunction(initial_theta, mappedFeatures, y))

    theta, minimumCost = optimizeTheta(initial_theta, mappedFeatures, y)

    # ***************************************** Step 4: Results *****************************************

    print("\n\t __________________________ Results __________________________")

    print("\n\tOptimal theta values: ", theta)
    print("\n\tMinimum cost value: ", minimumCost)



    # # ***************************************** Step 4: Plot decision boundaries *****************************************

    plotDifferentBoundaries(theta, mappedFeatures, y, positiveExamples, negativeExamples)

    print("\n******************* Exiting **********************")


if __name__ == '__main__':
    main()