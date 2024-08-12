import pandas as pd
import numpy as np
# from pathlib import Path 
import io
import dvc.api, dvc.repo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score
import mlflow
import mlflow.sklearn
import os
from datetime import datetime
# import joblib
import warnings

warnings.filterwarnings('ignore')

url = "https://github.com/mostafa-fallaha/data-drift-simulation"
data = dvc.api.read("new_data/Google-Playstore.csv", encoding='utf-8', repo=url)
df = pd.read_csv(io.StringIO(data))

df_A = df.loc[(df['Released_Year'] >= 2010) & (df['Released_Year'] <= 2017)]

# Step 1: Convert boolean columns to 0 and 1
df_A['Free'] = df_A['Free'].replace({True: 1, False: 0})
df_A['Ad Supported'] = df_A['Ad Supported'].replace({True: 1, False: 0})
df_A['In App Purchases'] = df_A['In App Purchases'].replace({True: 1, False: 0})
df_A['Editors Choice'] = df_A['Editors Choice'].replace({True: 1, False: 0})

# Step 2: Convert categorical variables into dummy variables
df_A = pd.get_dummies(df_A, columns=['Category', 'Content Rating'])

# Step 3: Feature Scaling
scaler = StandardScaler()

# Selecting numerical columns for scaling
numerical_columns = ['Rating', 'Rating Count', 'Installs', 'Price', 'Size_M', 'Released_Year', 'Released_Month', 'Days_Between']
df_A[numerical_columns] = scaler.fit_transform(df_A[numerical_columns])


X = df_A[df_A.columns.difference(['Daily_Avg_Installs'])]
y = df_A.Daily_Avg_Installs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

polynomial = PolynomialFeatures(degree=2, include_bias= False, interaction_only = False)
X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.transform(X_test)

linear_pol_model = LinearRegression()
linear_pol_model.fit(X_train_poly, y_train)
pred_poly = linear_pol_model.predict(X_test_poly)

rmse = root_mean_squared_error(y_test, pred_poly)
r2 = r2_score(y_test, pred_poly)
print(rmse)
print(r2)

# ============================== Logging the model with MLFlow============================================================
script_path = os.path.abspath(__file__)
runname = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Start MLflow run
mlflow.set_experiment('google_apps_model')
with mlflow.start_run(run_name=runname) as mlflow_run:
    run_id = mlflow_run.info.run_id

    # Log model parameters
    mlflow.log_param("degree", 2)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)


    # Log the model
    mlflow.sklearn.log_model(linear_pol_model, "model")

    # Register the model
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="linear_pol_model_a")
    
    # Log artifacts
    mlflow.log_artifact(script_path)