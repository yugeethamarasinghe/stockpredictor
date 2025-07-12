import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib
matplotlib.use('Agg')  # Add this line before importing matplotlib.pyplot
import matplotlib.pyplot as plt

# Lists to hold dates and prices
dates = []
prices = []

# Function to load data from CSV
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        header = next(csvFileReader)  # Get header row
        date_index = header.index("Date")
        open_index = header.index("Open")
        for row in csvFileReader:
            dates.append(int(row[date_index].split('/')[1]))  # Extract day from date (MM/DD/YYYY)
            prices.append(float(row[open_index]))  # Use 'Open' price

# Function to predict prices using SVR
def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Day of Month')
    plt.ylabel('Open Price')
    plt.title('NVDA Stock Price Prediction Using SVR')
    plt.legend()
    plt.savefig("stock_prediction.png")

    x = np.array([[x]])  # Reshape x for prediction
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

# Load your data from CSV (replace with your NVDA stock CSV filename)
get_data('nvda.csv')

# Predict price for a specific date (e.g., day = 29)
predicted_price = predict_prices(dates, prices, 29)

# Print the predicted prices
print(predicted_price)
