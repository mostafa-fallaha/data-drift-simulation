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

df_A_B = df.loc[(df['Released_Year'] >= 2010) & (df['Released_Year'] <= 2019)]

# Step 1: Convert boolean columns to 0 and 1
df_A_B['Free'] = df_A_B['Free'].replace({True: 1, False: 0})
df_A_B['Ad Supported'] = df_A_B['Ad Supported'].replace({True: 1, False: 0})
df_A_B['In App Purchases'] = df_A_B['In App Purchases'].replace({True: 1, False: 0})
df_A_B['Editors Choice'] = df_A_B['Editors Choice'].replace({True: 1, False: 0})

# Step 2: Convert categorical variables into dummy variables
df_A_B = pd.get_dummies(df_A_B, columns=['Category', 'Content Rating'])

# Step 3: Feature Scaling
scaler = StandardScaler()

# Selecting numerical columns for scaling
numerical_columns = ['Rating', 'Rating Count', 'Installs', 'Price', 'Size_M', 'Released_Year', 'Released_Month', 'Days_Between']
df_A_B[numerical_columns] = scaler.fit_transform(df_A_B[numerical_columns])


X = df_A_B[df_A_B.columns.difference(['Daily_Avg_Installs'])]
y = df_A_B.Daily_Avg_Installs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)



# polynomial = PolynomialFeatures(degree=2, include_bias= False, interaction_only = False)
# X_train_poly = polynomial.fit_transform(X_train)
# X_test_poly = polynomial.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
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
    mlflow.sklearn.log_model(linear_model, "model")

    # Register the model
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="linear_model_a")
    
    # Log artifacts
    mlflow.log_artifact(script_path)