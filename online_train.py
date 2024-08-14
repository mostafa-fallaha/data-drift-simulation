import pandas as pd
import dvc.api, dvc.repo
import io
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

url = "https://github.com/mostafa-fallaha/data-drift-simulation"
data = dvc.api.read("data/new.csv", encoding='utf-8', repo=url)
df = pd.read_csv(io.StringIO(data))

df2 = pd.read_csv("data/model_data.csv")

combined = pd.concat([df, df2, df2]).drop_duplicates(keep=False)
df_diff = combined

print("new data:",df.shape)
print("memory data:",df2.shape)
print("difference:",df_diff.shape)

# Step 1: Convert boolean columns to 0 and 1
df_diff['Free'] = df_diff['Free'].replace({True: 1, False: 0})
df_diff['Ad Supported'] = df_diff['Ad Supported'].replace({True: 1, False: 0})
df_diff['In App Purchases'] = df_diff['In App Purchases'].replace({True: 1, False: 0})
df_diff['Editors Choice'] = df_diff['Editors Choice'].replace({True: 1, False: 0})

# Step 2: Convert categorical variables into dummy variables
df_diff = pd.get_dummies(df_diff, columns=['Category', 'Content Rating'])

# Step 3: Feature Scaling
scaler = StandardScaler()

# Selecting numerical columns for scaling
numerical_columns = ['Rating', 'Rating Count', 'Installs', 'Price', 'Size_M', 'Released_Year', 'Released_Month', 'Days_Between']
df_diff[numerical_columns] = scaler.fit_transform(df_diff[numerical_columns])


X = df_diff[df_diff.columns.difference(['Daily_Avg_Installs'])]
y = df_diff.Daily_Avg_Installs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

model = SGDRegressor(max_iter=1000, tol=1e-3)
model.partial_fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("rmse:",rmse)
print("r2:",r2)

filepath = Path('data/model_data.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)