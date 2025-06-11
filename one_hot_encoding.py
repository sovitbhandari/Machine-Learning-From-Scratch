##Note: One-hot encoding is a method to convert categorical values into binary columns — one 
#for each category — where only the matching category gets a 1, and all others get 0.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

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

# columnd that need to one-hot encoded
low_cardinality_cols = [col for col in categorial_col if X_train[col].nunique() < 10]

#Columns to be dropped
high_cardinality_cols = list(set(categorial_col) - set(low_cardinality_cols))

## dropping high-cardinality-columns
label_X_train = X_train.drop(high_cardinality_cols, axis= 1)
label_X_valid = X_valid.drop(high_cardinality_cols, axis= 1)

## One-Hot Encoding
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_X_train = pd.DataFrame(OH_encoder.fit_transform(label_X_train[low_cardinality_cols]))
OH_X_valid = pd.DataFrame(OH_encoder.transform(label_X_valid[low_cardinality_cols]))

# Restore index to align one-hot encoded columns with the original training data
OH_X_train.index = label_X_train.index
OH_X_valid.index = label_X_valid.index

# Remove original categorical columns before concat
num_X_train = label_X_train.drop(low_cardinality_cols, axis=1)
num_X_valid = label_X_valid.drop(low_cardinality_cols, axis=1)

Final_X_train = pd.concat([num_X_train, OH_X_train], axis = 1)
Final_X_valid = pd.concat([num_X_valid, OH_X_valid], axis = 1)

Final_X_train.columns = Final_X_train.columns.astype(str)
Final_X_valid.columns = Final_X_valid.columns.astype(str)


print("Approach 3's MAE (One-Hot Encoding):") 
print(score_dataset(Final_X_train, Final_X_valid, y_train, y_valid))