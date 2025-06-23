# Ford Car Price Recommender System

The Ford Car Price Recommender System is a Flask-based web application that predicts the price of Ford cars using multiple machine learning models. Built with a modern, responsive UI, it allows users to input car details (model, year, transmission, mileage, fuel type, tax, MPG, and engine size) and receive price predictions from Linear Regression, Decision Tree, Polynomial Regression, and Neural Network models. The app is deployed on Render for easy access and leverages scikit-learn and TensorFlow for model training and inference.

## Features
- **Multiple Models**: Predicts car prices using four machine learning models:
  - Linear Regression
  - Decision Tree Regressor
  - Polynomial Regression (degree 3)
  - Artificial Neural Network (ANN) with TensorFlow
- **Modern UI**: Clean, responsive design with a card-based layout, smooth animations, and Google Fonts (Inter) for a professional look.
- **User Input**: Intuitive form for entering car details, with dropdowns for categorical features and number inputs for numerical features.
- **Preprocessing**: Handles numerical scaling (`MinMaxScaler`) and categorical encoding (`OneHotEncoder` with `handle_unknown='ignore'`).
- **Deployment**: Hosted on Render for scalable, cloud-based access.
- **Responsive Design**: Optimized for desktops, tablets, and mobile devices.

## Project Structure

Ford-Car-Price-Recommender-System/├── app.py                   # Flask application code├── requirements.txt         # Python dependencies├── preprocessor.pkl         # Preprocessor for Linear Regression, Decision Tree, and ANN├── poly_preprocessor.pkl    # Preprocessor for Polynomial Regression├── model_lr.pkl            # Linear Regression model├── model_dt.pkl            # Decision Tree model├── model_poly.pkl          # Polynomial Regression model├── model_ann.h5            # Neural Network model (HDF5 format)├── static/│   └── style.css           # Modern CSS styles├── templates/│   ├── index.html          # Home page with input form│   └── result.html         # Results page with predictions└── .gitignore              # Git ignore file for excluding virtual env, etc.

## Installation
Follow these steps to set up the project locally.

### Prerequisites
- Python 3.10
- Git
- PyCharm or another IDE (optional)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RDX-Ali-Ansari/Ford-Car-Price-Recommender-System.git
   cd Ford-Car-Price-Recommender-System


Set Up a Virtual Environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Ensure requirements.txt includes:
flask
scikit-learn==1.7.0
tensorflow
pandas
numpy
gunicorn


Verify Model Files:

Ensure preprocessor.pkl, poly_preprocessor.pkl, model_lr.pkl, model_dt.pkl, model_poly.pkl, and model_ann.h5 are in the project root.
If missing, generate them using save_models_updated.py (requires ford.csv dataset):python save_models_updated.py




Run the App Locally:
python app.py


Alternatively, use Gunicorn:gunicorn app:app


Open http://127.0.0.1:5000 in a browser.



Usage

Access the App:

Locally: Navigate to http://127.0.0.1:5000.
Deployed: Visit the Render URL (e.g., https://ford-car-price-predictor.onrender.com).


Input Car Details:

Fill out the form with:
Model: Select a Ford model (e.g., Fiesta, Focus).
Year: Choose a year (1996–2020).
Transmission: Select Manual, Automatic, or Semi-Auto.
Fuel Type: Choose Petrol, Diesel, or Hybrid.
Mileage: Enter miles (e.g., 20000).
Tax: Enter tax in pounds (e.g., 150).
MPG: Enter miles per gallon (e.g., 50.0).
Engine Size: Enter engine size in liters (e.g., 1.0).


Click Predict Price.


View Predictions:

The results page displays predicted prices from all four models and the input data for reference.



Model Details

Dataset: Ford car price dataset (ford.csv), containing features like model, year, transmission, mileage, fuel type, tax, MPG, and engine size.
Preprocessing:
Numerical features (mileage, tax, mpg, engineSize) scaled with MinMaxScaler.
Categorical features (model, transmission, fuelType) encoded with OneHotEncoder (drops first category, ignores unknown categories).
Outliers removed (e.g., year != 2060, fuelType != 'Other').


Models:
Linear Regression: Simple linear model for baseline predictions.
Decision Tree: Captures non-linear relationships.
Polynomial Regression: Degree-3 polynomial features for enhanced fit.
Neural Network: 3-layer ANN (64, 32, 16 neurons) with ReLU activation and Huber loss.



Deployment on Render
The app is deployed on Render for cloud access. Follow these steps to deploy your own instance:

Push to GitHub:
git add .
git commit -m "Update project files"
git push origin main


Set Up Render:

Sign in to render.com with GitHub.
Create a Web Service, select Ford-Car-Price-Recommender-System, and configure:
Branch: main
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
Environment Variables:
PYTHON_VERSION=3.10
TF_ENABLE_ONEDNN_OPTS=0 (optional)


Instance Type: Free (or paid for production).




Deploy:

Click Create Web Service.
Access the app at the provided URL (e.g., https://ford-car-price-predictor.onrender.com).



Troubleshooting

Unknown Category Error:
Ensure preprocessor.pkl and poly_preprocessor.pkl use handle_unknown='ignore' in save_models_updated.py.
Regenerate models and push to GitHub.


Build Fails on Render:
Verify requirements.txt includes all dependencies.
Check file sizes for .pkl and .h5 (use cloud storage if >100MB).


Local Run Issues:
Confirm Python 3.10 and scikit-learn 1.7.0 are installed.
Run pip install -r requirements.txt again.



Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.
