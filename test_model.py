from sklearn.linear_model import LinearRegression
import numpy as np

def test_prediction():

    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[4]])

    assert round(prediction[0]) == 8