# [Kaggle Machine Learning Tutorial](https://www.kaggle.com/learn/machine-learning)

## Setup
import pandas as pd
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
# print(data.describe())
# print(data.columns)
# print(data.SalePrice.head())

## Decision Tree Model
y = data.SalePrice
x_factors = ['YrSold', 'LotArea', 'YearBuilt', 'OverallCond', 'OverallQual', '1stFlrSF', '2ndFlrSF']
their_x_factors = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
X = data[x_factors]
# print(X.describe())

from sklearn.tree import DecisionTreeRegressor
house_model = DecisionTreeRegressor()
house_model.fit(X, y)
predicted_home_prices = house_model.predict(X)

## Model Validation (https://www.kaggle.com/dansbecker/model-validation)
from sklearn.metrics import mean_absolute_error
error1 = mean_absolute_error(y, predicted_home_prices)
# print("error1 (inacurate, missing train_test_split): ", error1)

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
house_model_v2 = DecisionTreeRegressor()
house_model_v2.fit(train_X, train_y)

val_predictions = house_model_v2.predict(val_X)
error2 = mean_absolute_error(val_y, val_predictions)
print("DecisionTreeRegressor mean_absolute_error: ", error2)

## [Underfitting, Overfitting, & MOdel Optimiation](https://www.kaggle.com/dansbecker/underfitting-overfitting-and-model-optimization)
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("DecisionTreeRegressor - Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
    
## [Random Forests](https://www.kaggle.com/dansbecker/random-forests)
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
print("RandomForestRegressor - mean_absolute_error: ", mean_absolute_error(val_y, forest_preds))

## [Submitting from a Kernel](https://www.kaggle.com/dansbecker/submitting-from-a-kernel/)
# Read the data
train = pd.read_csv('../input/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
# predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
predictor_cols = ['YrSold', 'LotArea', 'YearBuilt', 'OverallCond', 'OverallQual', '1stFlrSF', '2ndFlrSF']

# Create training predictors data
train_X = train[predictor_cols]
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
# print(predicted_prices)

## Write Submission
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
