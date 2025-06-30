import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

X = pd.read_csv('data-files/train.csv', index_col='Id')
X_test_full= pd.read_csv('data-files/test.csv', index_col='Id')

X.dropna(axis = 0, subset = ["SalePrice"], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis = 1, inplace= True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == "object"]

#numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Defining the model
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05) # Your code here

# Fitting the model
my_model.fit(X_train, y_train) # Your code here

predictions_1 = my_model.predict(X_valid)

mae_1 = mean_absolute_error(predictions_1, y_valid)

print("Mean Absolute Error:" , mae_1)