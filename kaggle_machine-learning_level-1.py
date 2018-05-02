## Setup
import pandas as pd
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
# print(data.describe())
print(data.columns)
# print(data.SalePrice.head())

## Decision Tree Model
y = data.SalePrice
x_factors = ['YrSold', 'LotArea', 'YearBuilt', 'OverallCond', 'OverallQual', '1stFlrSF', '2ndFlrSF']
X = data[x_factors]
# print(X.describe())

from sklearn.tree import DecisionTreeRegressor
house_model = DecisionTreeRegressor()
house_model.fit(X, y)
predicted_home_prices = house_model.predict(X)

## Model Validation (https://www.kaggle.com/dansbecker/model-validation)
from sklearn.metrics import mean_absolute_error
error1 = mean_absolute_error(y, predicted_home_prices)
print("error1 (inacurate, missing train_test_split): ", error1)

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
house_model_v2 = DecisionTreeRegressor()
house_model_v2.fit(train_X, train_y)

val_predictions = house_model_v2.predict(val_X)
error2 = mean_absolute_error(val_y, val_predictions)
print("error2: ", error2)

## [Underfitting, Overfitting, & MOdel Optimiation](https://www.kaggle.com/dansbecker/underfitting-overfitting-and-model-optimization)
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
    
## [Random Forests](https://www.kaggle.com/dansbecker/random-forests)
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
print("forest_preds - mean_absolute_error: ", mean_absolute_error(val_y, forest_preds))
