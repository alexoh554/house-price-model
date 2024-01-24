import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from data import load_data

def main():
    x, y = load_data()

    # Scale features
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)

    # Perform gradient descent 
    sgdr = SGDRegressor(max_iter = 10000)
    sgdr.fit(x_norm, y)

    # Get parameters of linear equation
    b = sgdr.intercept_
    w = sgdr.coef_
    print(f"model parameters: w= {w}, b= {b}")



if __name__ == "__main__":
    main()
