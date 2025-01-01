## About
Some experiemnting I did with machine learning after taking DeepLearning.Ai's [Machine Learning Specialization course](https://github.com/alexoh554/house-price-model/blob/main/Regression%20and%20Classification%20Course%20Certificate.pdf).

The goal was to take housing price data from Kaggle and create a linear regression model to predict a house's market price based on features such as square feet, # of bathrooms, year built, and more. 

## Models
`manual_gradient_descent.py`
- In this file, I manually implemented the gradient descent function using weight and bias formulas
- I also manually implemented the cost function using the cost function formula
- Uses Z-score normalization

`linear_regression_sklearn.py`
- Much more condensed and customizable implementation thanks to Scikit-learn's `SGDRegressor` class
