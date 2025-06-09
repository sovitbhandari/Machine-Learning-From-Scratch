import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

home_data = pd.read_csv('data-files/train.csv')
home_test_data = pd.read_csv('data-files/test.csv')

X = home_data.select_dtypes(exclude = ['object'])
X_test = home_test_data.select_dtypes(exclude=['object'])

y = home_data.SalePrice


def score_dataset(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_X, train_y)
    predicted_val = model.predict(val_X)
    return mean_absolute_error(val_y, predicted_val)


train_X, val_X, train_y, val_y = train_test_split(X, y, train_size= 0.8, test_size = 0.2, random_state = 0)

train_X_new = train_X.copy()
val_X_new = val_X.copy()

missing_cols = [col for col in train_X if train_X[col].isnull().any()]

for col in missing_cols:
    train_X_new[col + '_was_missing'] = train_X_new[col].isnull()
    val_X_new[col + '_was_missing'] = val_X_new[col].isnull()


imputer = SimpleImputer()
imputed_train_X_new = pd.DataFrame(imputer.fit_transform(train_X_new))
imputed_val_X_new = pd.DataFrame(imputer.transform(val_X_new))

# Restore original column names after imputation
imputed_train_X_new.columns = train_X_new.columns
imputed_val_X_new.columns = val_X_new.columns

print("The extended imputed MAE is: ", score_dataset(imputed_train_X_new, imputed_val_X_new, train_y, val_y))






