import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from data import load_data

def main():
    x, y, features= load_data()

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

    # Predict value of y using wx+b
    y_pred = sgdr.predict(x_norm)
    print(y_pred)

    # Compare target vs predicted results
    num_features = len(x[0])
    for i in range(num_features):
        fig, ax = plt.subplots()

        ax.scatter(x[:, i], y, label='Actual')
        ax.set_xlabel(features[i])

        ax.scatter(x[:, i], y_pred, color="orange", label='Predicted')
        ax.set_ylabel('Target')

        plt.show()


    # Get User feature input for prediction
    print("Enter attributes of the house: ") 
    user_features = []
    for feature in features:
        user_features.append(float(input(f"{feature} = ")))


    x_given = np.array(user_features).reshape(1, -1)
    x_given_norm = scaler.transform(x_given)

    user_predicted = sgdr.predict(x_given_norm)
    print(f"Predicted value of the house: ${user_predicted}")

if __name__ == "__main__":
    main()
