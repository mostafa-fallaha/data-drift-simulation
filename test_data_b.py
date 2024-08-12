import mlflow
import mlflow.sklearn
import pandas as pd
# import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model("models:/linear_model_a/latest")

df_b = pd.read_csv("new_data/df_B.csv")

# Step 1: Convert boolean columns to 0 and 1
df_b['Free'] = df_b['Free'].replace({True: 1, False: 0})
df_b['Ad Supported'] = df_b['Ad Supported'].replace({True: 1, False: 0})
df_b['In App Purchases'] = df_b['In App Purchases'].replace({True: 1, False: 0})
df_b['Editors Choice'] = df_b['Editors Choice'].replace({True: 1, False: 0})

# Step 2: Convert categorical variables into dummy variables
df_b = pd.get_dummies(df_b, columns=['Category', 'Content Rating'])

# Step 3: Feature Scaling
scaler = StandardScaler()

# Selecting numerical columns for scaling
numerical_columns = ['Rating', 'Rating Count', 'Installs', 'Price', 'Size_M', 'Released_Year', 'Released_Month', 'Days_Between']
df_b[numerical_columns] = scaler.fit_transform(df_b[numerical_columns])

X = df_b[df_b.columns.difference(['Daily_Avg_Installs'])]
y = df_b.Daily_Avg_Installs

pred_b = model.predict(X)

rmse = root_mean_squared_error(y, pred_b)
r2 = r2_score(y, pred_b)
print(rmse)
print(r2)

# there's an error, because after 2017 we got a new Content Rating which is Adults only 18+