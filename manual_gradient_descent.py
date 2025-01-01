import numpy as np
import matplotlib.pyplot as plt
import copy
from data import load_data

def main():
    x, y = load_data()

    # Scale features 
    x_n = normalize(x)

    w_initial = np.zeros((2,)) 
    b_initial = 0

    w, b, J_history = gradient_descent(x_n, y, w_initial, b_initial, 0.001, 100000)

    fig, ax = plt.subplots()
    ax.scatter(J_history, np.arange(len(J_history)))
    plt.show()



def gradient_descent(x, y, w_in, b_in, alpha, max_iter):
    """
    Repeat until convergence/max iterations:
    w = w - dj_dw * alpha
    b = b - dj_db * alpha
    """
    m, n = x.shape
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    # For graphing use
    J_history = []

    for i in range(max_iter):
        dj_dw, dj_db = compute_gradients(x, y, w, b)

        w -= dj_dw * alpha
        b -= dj_db * alpha

        J_history.append(compute_cost(x, y, w, b))

    return w, b, J_history
        
def compute_cost(x, y, w, b):
    """
    Formula:
    cost = ((f(x) - y)^2) / 2m
    """
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        f = np.dot(x[i],w) + b           
        cost += (f - y[i])**2
    cost = cost / (2*m)
    return cost 

def compute_gradients(x, y, w, b):
    """
    Formula:
    dj_dw = ((f(x) - y) * x ) / 2m
    dj_db = (f(x) - y) / 2m
    """
    m,n = x.shape           #(number of examples, number of features)

    dj_dw = np.zeros((n,))
    dj_db = 0.


    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        print(type(err))
        for j in range(n):
            dj_dw[j] += err * x[i,j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db,dj_dw


def normalize(x):
    """
    Normalize using z-score normalization:
    x_i = (x_i - mu_i) / sigma_i
    """
    mu = np.mean(x, axis = 0)

    # sigma = standard deviation
    sigma = np.std(x, axis = 0) 
    x_n = (x - mu) / sigma
    return x_n

if __name__ == "__main__":
    main()