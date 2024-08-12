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

df_c = pd.read_csv("new_data/df_C.csv")

# Step 1: Convert boolean columns to 0 and 1
df_c['Free'] = df_c['Free'].replace({True: 1, False: 0})
df_c['Ad Supported'] = df_c['Ad Supported'].replace({True: 1, False: 0})
df_c['In App Purchases'] = df_c['In App Purchases'].replace({True: 1, False: 0})
df_c['Editors Choice'] = df_c['Editors Choice'].replace({True: 1, False: 0})

# Step 2: Convert categorical variables into dummy variables
df_c = pd.get_dummies(df_c, columns=['Category', 'Content Rating'])

# Step 3: Feature Scaling
scaler = StandardScaler()

# Selecting numerical columns for scaling
numerical_columns = ['Rating', 'Rating Count', 'Installs', 'Price', 'Size_M', 'Released_Year', 'Released_Month', 'Days_Between']
df_c[numerical_columns] = scaler.fit_transform(df_c[numerical_columns])

X = df_c[df_c.columns.difference(['Daily_Avg_Installs'])]
y = df_c.Daily_Avg_Installs

pred_b = model.predict(X)

rmse = root_mean_squared_error(y, pred_b)
r2 = r2_score(y, pred_b)
print(rmse)
print(r2)

# there's an error, because after 2017 we got a new Content Rating which is Adults only 18+