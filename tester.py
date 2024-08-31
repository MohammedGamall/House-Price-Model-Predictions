import pandas as pd 
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score, classification_report

x_test = pd.read_csv('test.csv')
y_test = pd.read_csv('target_test.csv',usecols= ['price'])

rf = joblib.load('random_forest.pkl')

# Evaluate the ensemble model on the train set
y_pred = rf.predict(x_test)

# Calculate RMSE and R²  for train data
test_rmse= np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Regularized Train RMSE: {test_rmse:.4f}")
print(f"R²: {r2:.4f}")