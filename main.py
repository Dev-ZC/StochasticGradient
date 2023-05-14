import numpy as np

# My Artifical Data

true_weights = np.array([1, 2, 3, 4, 5])
dimensions = len(true_weights) # In this case 5 dimensions
points = []

for i in range(10000):
    x = np.random.randn(dimensions)
    y = true_weights.dot(x) + np.random.randn()
    print(x, y)
    points.append((x, y))
    
def squaredLoss(w, i):
    # Finds the uses the residual to find the squared loss 
    # of every point and adds them together, then divides them 
    # by len(points) to get the average
    x, y = points[i]
    return (w.dot(x) - y)**2

def squaredLossDerivative(w, i):
    x, y = points[i]
    return 2*(w.dot(x) - y) * x

def stochasticGradientDescent(squaredLoss, squaredLossDerivative, dimensions, n):
    # Fills a new area w with zeros 
    w = np.zeros(dimensions)
    stepSize = 1
    numUpdates = 0
    for t in range(1000):
        for i in range(n):
            value = squaredLoss(w, i)
            # Gradient is the direction that maximizes the loss
            gradient = squaredLossDerivative(w, i)
            numUpdates += 1
            stepSize = 1.0/numUpdates
            # Therefoe we multiply the gradient by our stepsize and subtract
            # it from our weight to get closer to the minimum amount of loss
            w = w - stepSize * gradient
        print('iteration {}: w = {}, squaredLoss(w) = {}'.format(t, w, value))
        
stochasticGradientDescent(squaredLoss, squaredLossDerivative, dimensions, len(points))
    