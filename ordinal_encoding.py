##Note: Ordinal encoding assigns each unique value to a different intege

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

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

## Categorical columns in the training data(X_train)
categorial_col = [col for col in X_train.columns if X_train[col].dtype == "object"]

## Select categorical columns where all validation values are present in training (safe for OrdinalEncoder)
good_ordinal_col = [col for col in categorial_col if set(X_valid[col]).issubset(set(X_train[col]))]

## Problematic columns to be dropped from the dataset
bad_ordinal_col = list(set(categorial_col) - set(good_ordinal_col))

#Dropping Bad Ordinal columns from both X_train and X_valid
label_X_train = X_train.drop(bad_ordinal_col, axis=1)
label_X_valid = X_valid.drop(bad_ordinal_col, axis=1)


# Applying ordinal encoder
ordinal_encoder = OrdinalEncoder()
label_X_train[good_ordinal_col] = ordinal_encoder.fit_transform(X_train[good_ordinal_col])
label_X_valid[good_ordinal_col] = ordinal_encoder.transform(X_valid[good_ordinal_col])

print("Approach 2's MAE (Ordinal Encoding): ")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))