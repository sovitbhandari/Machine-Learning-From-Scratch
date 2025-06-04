import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


data_path = 'data-files/train.csv'
read_data = pd.read_csv(data_path)

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = read_data[feature_names]
y = read_data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state= 1)

random_forest_model = RandomForestRegressor(random_state = 1)

random_forest_model.fit(train_X, train_y)
Predicted_val = random_forest_model.predict(val_X)

print("Predicted Sale Prices for Validation Examples: ", random_forest_model.predict(val_X.head()).tolist())

MAE = mean_absolute_error(val_y, Predicted_val)

print("Mean Absolute error: ", MAE)