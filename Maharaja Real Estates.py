#!/usr/bin/env python
# coding: utf-8

## Maharaja Real Estates - Price Predictor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Load and inspect the data
housing = pd.read_csv("data.csv")
housing.info()
housing['CHAS'].value_counts()
housing.describe()

# Train-Test Splitting using Scikit-Learn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

# Looking for Correlations
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# Attribute Combinations
housing['TAXRM'] = housing['TAX'] / housing['RM']
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# Prepare the data for Machine Learning algorithms
housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()

# Handle Missing Attributes
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)

# Creating a Pipeline
my_pipeline = Pipeline([
    ('Imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)

# Selecting a desired model for Maharaja Real Estates
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# Evaluating the model
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# Using better evaluation technique - Cross Validation
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

print_scores(rmse_scores)

# Saving the Model
dump(model, 'Maharaja.joblib')

# Testing the model on test data
X_test = strat_test_set.drop('MEDV', axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))
print(final_rmse)

# Using the model
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)