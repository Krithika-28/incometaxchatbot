import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import numpy as np

# Load dataset
df = pd.read_csv('D:/Project/myenv/India_Income_Tax_Calculation.csv')

# Use only Salary as input
X = df[['Salary']]
y = df['Total Tax Payable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Model Evaluation:")
print(f"🔹 R² Score       : {r2:.4f}")
print(f"🔹 RMSE (₹)       : {rmse:.2f}")

# Save the trained model
with open("D:/Project/myenv/tax_model.pkl", "wb") as f:
    pickle.dump(model, f)
