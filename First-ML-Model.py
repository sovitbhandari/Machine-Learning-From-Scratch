import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data_path = 'train.csv'
read_data = pd.read_csv(data_path)
# print(read_data.columns)

'''Specify Prediction Target'''
y = read_data.SalePrice

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = read_data[feature_names]


'''model reproducibility'''
decision_model = DecisionTreeRegressor(random_state = 1)

'''Fit the model'''
decision_model.fit(X,y)

'''Model's prediction'''
print("Making Predictions for first 5 houses:")
print(X.head())
print("The predicted prices are")
print(decision_model.predict(X.head()))