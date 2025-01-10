import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load datasets
col_names_1 = ["tech_prev","entertain_prev","tech_next","entertain_next"]
col_name_2 = ["tech", "entertain"]
historical_data = pd.read_csv('data.csv', header=None, names= col_names_1)
current_creators = pd.read_csv('creators.csv', header=None, names= col_name_2)

# Preprocess historical data

X = historical_data[['tech_prev', 'entertain_prev']].values
y_tech = historical_data['tech_next'].values
y_entertain = historical_data['entertain_next'].values

# Train models
tech_model = LinearRegression().fit(X, y_tech)
entertain_model = LinearRegression().fit(X, y_entertain)

# Predict for 4 weeks
predicted_creators = current_creators.copy()
for week in range(4):
    X_current = predicted_creators[['tech', 'entertain']].values
    predicted_creators['tech'] = tech_model.predict(X_current)
    predicted_creators['entertain'] = entertain_model.predict(X_current)

# Find results
highest_tech = predicted_creators['tech'].idxmax()
highest_entertain = predicted_creators['entertain'].idxmax()

# Detect focus switching
initial_focus = current_creators['tech'] > current_creators['entertain']
final_focus = predicted_creators['tech'] > predicted_creators['entertain']
switched_to_entertain = np.where((~initial_focus) & final_focus)[0]
switched_to_tech = np.where(initial_focus & (~final_focus))[0]

# Output results
print("Highest tech index:", highest_tech)
print("Highest entertain index:", highest_entertain)
print("Switched to entertain:", switched_to_entertain)
print("Switched to tech:", switched_to_tech)
