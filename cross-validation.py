
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('data-files/train.csv', index_col='Id')
test_data = pd.read_csv('data-files/test.csv', index_col='Id')

#remove rows with missing value in SalePrice
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice

#Drop SalePrice from X because it is used to predict price and uses feature
train_data.drop(['SalePrice'], axis = 1, inplace=True)

#Selecting numerical cols
numerical_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]
X = train_data[numerical_cols].copy()
X_test = test_data[numerical_cols].copy()

def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])
    
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

plt.plot(list(results.keys()), list(results.values()))
plt.show()


n_estimators_best = min(results, key=results.get)

print(n_estimators_best)