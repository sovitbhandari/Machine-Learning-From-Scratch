import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

home_data = pd.read_csv('data-files/train.csv')
home_test_data = pd.read_csv('data-files/test.csv')

X = home_data.select_dtypes(exclude = ['object'])
X_test = home_test_data.select_dtypes(exclude = ['object'])

y = home_data.SalePrice


train_X, val_X, train_y, val_y = train_test_split(X, y, train_size= 0.8, test_size = 0.2, random_state = 0)


def score_dataset(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_X, train_y)
    predicted_val = model.predict(val_X)
    return mean_absolute_error(val_y, predicted_val)


missing_cols = [col for col in train_X if train_X[col].isnull().any()]

reduced_train_X = train_X.drop(missing_cols, axis = 1)
reduced_val_X = val_X.drop(missing_cols, axis = 1)

print("MAE on dropping columns: ",score_dataset(reduced_train_X, reduced_val_X, train_y, val_y))