import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

home_data = pd.read_csv('data-files/train.csv')
home_test_data = pd.read_csv('data-files/test.csv')

X = home_data.select_dtypes(exclude = ['object'])


y = home_data.SalePrice


train_X, val_X, train_y, val_y = train_test_split(X, y, train_size= 0.8, test_size = 0.2, random_state = 0)


def score_dataset(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_X, train_y)
    predicted_val = model.predict(val_X)
    return mean_absolute_error(val_y, predicted_val)



imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(imputer.transform(val_X))

# Restore original column names after imputation
imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns

print("The imputed MAE is: ", score_dataset(imputed_train_X, imputed_val_X, train_y, val_y))






