import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


data_path = 'data-files/train.csv'
read_data = pd.read_csv(data_path)

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = read_data[feature_names]
y = read_data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state= 1)


#Utility function to find MAE(Mean Absolute Error) based on different leaf nodes
def get_MAE(maximum_leaf_nodes, train_X, val_X, train_y, val_y):
    decision_model = DecisionTreeRegressor(max_leaf_nodes= maximum_leaf_nodes, random_state = 1)
    decision_model.fit(train_X, train_y)
    val_prediction = decision_model.predict(val_X)
    MAE = mean_absolute_error(val_y, val_prediction)   # actual untrained price - predicted price from training set
    return(MAE)


max_leaf_options = [5, 25, 50, 100, 250, 500]
mae_dict = {} 

for max_leaf in max_leaf_options:
    current_mae = get_MAE(max_leaf, train_X, val_X, train_y, val_y)
    mae_dict[max_leaf] = current_mae
    print("For max_leaf_nodes = {:<4} → Mean Absolute Error = {:,}".format(max_leaf, current_mae))

best_tree_size = min(mae_dict, key=mae_dict.get)
print("\nBest max_leaf_nodes =", best_tree_size, "with Mean Absolute Error =", format(mae_dict[best_tree_size], ","))



# Fit final model on all data using the best tree size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
final_model.fit(X, y)  # use all available data


# Save the trained model
# joblib.dump(final_model, 'final_decision_tree_model.pkl')