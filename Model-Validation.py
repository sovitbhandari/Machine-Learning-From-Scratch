import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


data_path = 'data-files/train.csv'
read_data = pd.read_csv(data_path)

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = read_data[feature_names]
y = read_data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state= 1)

decision_model = DecisionTreeRegressor(random_state = 1)

decision_model.fit(train_X, train_y)

# The code above trains the actual model using training data.
# To make predictions on new data, we can use: decision_model.predict(<your data>).


# Code below validates the model using the unused portion of the original dataset (validation set)
# to check how well it performs by calculating the prediction error.


val_prediction = decision_model.predict(val_X)

print("Predicted Sale Prices for Validation Examples by Model: ", val_prediction[:5])

# Top few validation 
print("Actual Sale Prices of Validation Examples:", val_y.head().tolist())


MAE = mean_absolute_error(val_y, val_prediction)

print("Mean Absolute error: ", MAE)