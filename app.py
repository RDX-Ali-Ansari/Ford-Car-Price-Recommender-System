from flask import Flask, render_template, request
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and preprocessors
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('poly_preprocessor.pkl', 'rb') as f:
    poly_preprocessor = pickle.load(f)
with open('model_lr.pkl', 'rb') as f:
    lr_pipeline = pickle.load(f)
with open('model_dt.pkl', 'rb') as f:
    dt_pipeline = pickle.load(f)
with open('model_poly.pkl', 'rb') as f:
    poly_pipeline = pickle.load(f)
ann_model = load_model('model_ann.h5')

# Define unique values for dropdowns (based on dataset)
models = ['Fiesta', 'Focus', 'Kuga', 'EcoSport', 'C-MAX', 'Ka+', 'B-MAX', 'S-MAX', 'Mondeo', 'Edge', 'Mustang', 'Tourneo Custom', 'Grand C-MAX', 'Tourneo Connect', 'Galaxy', 'Puma', 'Grand Tourneo Connect', 'Fusion', 'Streetka', 'Ranger', 'Escort', 'Transit Tourneo', 'Cougar']
years = [str(year) for year in range(1996, 2021)]
transmissions = ['Manual', 'Automatic', 'Semi-Auto']
fuel_types = ['Petrol', 'Diesel', 'Hybrid']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=models, years=years, transmissions=transmissions, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'model': [request.form['model']],
        'year': [request.form['year']],
        'transmission': [request.form['transmission']],
        'mileage': [float(request.form['mileage'])],
        'fuelType': [request.form['fuelType']],
        'tax': [float(request.form['tax'])],
        'mpg': [float(request.form['mpg'])],
        'engineSize': [float(request.form['engineSize'])]
    }
    input_df = pd.DataFrame(data)

    # Preprocess and predict
    X_scaled = preprocessor.transform(input_df)
    lr_pred = lr_pipeline.predict(input_df)[0]
    dt_pred = dt_pipeline.predict(input_df)[0]
    poly_pred = poly_pipeline.predict(input_df)[0]
    ann_pred = ann_model.predict(X_scaled, verbose=0)[0][0]

    # Round predictions
    predictions = {
        'Linear Regression': round(lr_pred, 2),
        'Decision Tree': round(dt_pred, 2),
        'Polynomial Regression': round(poly_pred, 2),
        'Neural Network': round(ann_pred, 2)
    }

    return render_template('result.html', predictions=predictions, input_data=data)

if __name__ == '__main__':
    app.run(debug=True)