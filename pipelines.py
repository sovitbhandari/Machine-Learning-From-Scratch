
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

## Categorical columns in the training data(X_train)
categorial_col = [col for col in X_train.columns if X_train[col].dtype == "object"]

# columnd that need to one-hot encoded
low_cardinality_cols = [col for col in categorial_col if X_train[col].nunique() < 10]

#Selecting numerical cols
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]



#Combining Numerical columns and low cardinality columns
my_cols = low_cardinality_cols + numerical_cols


# Now getting new data table or set based on my new columns
X_train_new = X_train[my_cols].copy()
X_valid_new = X_valid[my_cols].copy()
X_test_new = X_test[my_cols].copy()

#numerical transformer
numerical_transformer = SimpleImputer(strategy='constant')

#preprocessing for low_cardinality categorical columns
categorial_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#preprocessing both numerical and categorical data
pre_processor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('categorical', categorial_transformer, low_cardinality_cols)])


#Defining Model
model = RandomForestRegressor(n_estimators=100, random_state=0)

#Bundle Preprocessing and modeling code in pipeline

pre_model_bundle = Pipeline(steps=[('preprocessor', pre_processor), ('model', model)])

pre_model_bundle.fit(X_train_new, y_train)

prediction = pre_model_bundle.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, prediction))