import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib



# Loading dataset
Med_Dataset = pd.read_csv("C:/Users/owner/PycharmProjects/MedProject/Medicine_Details.csv")

print("\n======Data Information======")
print(Med_Dataset.info())

print("\n======Data Head======")
print(Med_Dataset.head())

# Data preprocessing

# Excellent Review % column summary
Med_Dataset["Excellent Review %"].describe()

# Create a new column 'User_satisfaction' based on the values in 'Excellent Review %'
Med_Dataset['User_satisfaction'] = [
    'YES' if x > 39 else 'NO'
    for x in Med_Dataset['Excellent Review %']
]

print("\n======Data Head======")
print(Med_Dataset.head())

#Label Encoder of the Target Column
Med_Dataset['User_satisfaction']= Med_Dataset['User_satisfaction'].replace({'YES':1, 'NO':0})

print("\n======Data Head======")
print(Med_Dataset.head())


# Fill missing values with the mean (for numerical features) or the mode (for categorical features)
imputer = SimpleImputer(strategy='mean')
Med_Dataset[['Excellent Review %', 'Average Review %', 'Poor Review %']] = imputer.fit_transform(Med_Dataset[['Excellent Review %', 'Average Review %', 'Poor Review %']])

# Encode categorical columns using Label Encoding
label_encoders = {}
categorical_columns = ['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Image URL', 'Manufacturer']
for col in categorical_columns:
    le = LabelEncoder()
    Med_Dataset[col] = le.fit_transform(Med_Dataset[col])
    label_encoders[col] = le
print(le)

# Feature columns
X = Med_Dataset[['Excellent Review %','Average Review %', 'Poor Review %']]

# Target column
y = Med_Dataset['User_satisfaction']

# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# save the scaler
joblib.dump(scaler,'scaler.joblib')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Instantiate and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# saving the Random forest model
joblib.dump(rf_model, 'random_forest_model.joblib')

# Predict on the test set
y_pred = rf_model.predict(X_test)

 #Calculate the metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

