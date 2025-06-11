import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = pd.read_csv('data-files/train.csv', index_col='Id')
X_test = pd.read_csv('data-files/test.csv', index_col='Id')

#remove rows with missing value in SalePrice
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice

#Drop SalePrice from X because it is used to predict price and uses feature
X.drop(['SalePrice'], axis = 1, inplace=True)


#Drop columns that has missing values
cols_with_naValue = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_naValue, axis=1, inplace=True)
X_test.drop(cols_with_naValue, axis=1, inplace=True)


# Getting Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#Function to compare differrent approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    return mean_absolute_error(y_valid, prediction)

#Dropping Columns with categorical data
X_train_without_string_Dtype = X_train.select_dtypes(exclude=['object'])
X_valid_without_string_Dtype = X_valid.select_dtypes(exclude=['object'])

print("Approach 1's MAE (Drop categorical variables): ")
print(score_dataset(X_train_without_string_Dtype, X_valid_without_string_Dtype, y_train, y_valid))