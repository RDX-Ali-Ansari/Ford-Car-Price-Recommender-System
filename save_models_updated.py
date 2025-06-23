import pandas as pd
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the dataset path (update this to the correct path of ford.csv)
dataset_path = "ford.csv"  # e.g., "C:/Users/PMLS/Downloads/ford.csv"

# Check if the dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please update the dataset_path variable.")

# Load the dataset
df = pd.read_csv(dataset_path)

# Preprocess the data
df['year'] = df['year'].astype(str)
df = df.drop_duplicates()
df = df[df['year'] != '2060']
df = df[df['fuelType'] != 'Other']

# Define features
numerical_features = ['mileage', 'tax', 'mpg', 'engineSize']
categorical_features = ['model', 'transmission', 'fuelType']
X = df.drop(columns=['price'])
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)
# Fit preprocessor on entire X to capture all categories
preprocessor.fit(X)

# Polynomial preprocessor
poly_preprocessor = ColumnTransformer(
    transformers=[
        ('poly', Pipeline([('poly', PolynomialFeatures(degree=3)), ('scaler', MinMaxScaler())]), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)
# Fit poly_preprocessor on entire X
poly_preprocessor.fit(X)

# Linear Regression pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

# Decision Tree pipeline
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])
dt_pipeline.fit(X_train, y_train)

# Polynomial Regression pipeline
poly_pipeline = Pipeline(steps=[
    ('preprocessor', poly_preprocessor),
    ('model', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)

# ANN model
X_train_scaled = preprocessor.transform(X_train)  # Use transform, not fit_transform
input_dim = X_train_scaled.shape[1]
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=input_dim))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='Adam', loss='huber', metrics=['r2_score'])
model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=1)

# Save models and preprocessors
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('poly_preprocessor.pkl', 'wb') as f:
    pickle.dump(poly_preprocessor, f)
with open('model_lr.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)
with open('model_dt.pkl', 'wb') as f:
    pickle.dump(dt_pipeline, f)
with open('model_poly.pkl', 'wb') as f:
    pickle.dump(poly_pipeline, f)
model.save('model_ann.h5')

print("Models and preprocessors saved successfully.")