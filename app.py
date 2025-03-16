from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger
import zipfile
import io
import urllib.request

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Auto Trader Car Price Prediction API',
    'uiversion': 3}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cars.db'
db = SQLAlchemy(app)

# Define database model for cars
class Car(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    price = db.Column(db.Float, nullable=False)
    mileage = db.Column(db.Integer)
    year_of_registration = db.Column(db.Integer)
    standard_make = db.Column(db.String(50))
    standard_model = db.Column(db.String(50))

with app.app_context():
    db.create_all()

def preprocess_data(df, encoder=None, training=False):
    df = df.copy()
    
    # Clean numeric columns
    numeric_cols = ['price', 'mileage', 'year_of_registration']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Filter invalid data only during training
    if training:
        mask = (
            df['price'].notna() &
            (df['price'] > 0) & 
            (df['price'] < 1000000) &
            df['mileage'].notna() &
            (df['mileage'] >= 0) &
            df['year_of_registration'].notna() &
            (df['year_of_registration'] >= 1900))
        df = df[mask].copy()
    
    # Clean categorical columns
    categorical_cols = ['standard_make', 'standard_model']
    df[categorical_cols] = df[categorical_cols].fillna('Unknown').astype(str)
    
    # Create features
    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(df[categorical_cols])
    else:
        encoded_cats = encoder.transform(df[categorical_cols])
    
    # Create final features
    numeric_data = df[['mileage', 'year_of_registration']].values
    processed_data = np.column_stack((numeric_data, encoded_cats))
    
    return processed_data, encoder, df

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Load car listing data from CSV, process it, and train the model
    ---
    responses:
      200:
        description: Summary statistics of loaded data
    '''
    global model, encoder
    
    try:
        # Download and unzip data from GitHub
        url = 'https://github.com/DanaKChilds/airbnb/raw/refs/heads/master/AML_dataset.csv.zip'
        response = urllib.request.urlopen(url)
        zip_data = io.BytesIO(response.read())
        
        with zipfile.ZipFile(zip_data) as zip_file:
            csv_filename = zip_file.namelist()[0]  # Get the CSV filename from the zip
            with zip_file.open(csv_filename) as csv_file:
                cars_df = pd.read_csv(csv_file)
                
        # Verify required columns exist
        required_columns = ['price', 'mileage', 'year_of_registration', 
                          'standard_make', 'standard_model']
        missing_columns = [col for col in required_columns if col not in cars_df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {missing_columns}"}), 400

        # Convert price column if it contains currency symbols or commas
        if 'price' in cars_df.columns:
            cars_df['price'] = cars_df['price'].astype(str).str.replace('£','').str.replace(',','')
            cars_df['price'] = pd.to_numeric(cars_df['price'], errors='coerce')
            
        # Convert mileage to numeric
        if 'mileage' in cars_df.columns:
            cars_df['mileage'] = pd.to_numeric(cars_df['mileage'].astype(str).str.replace(r'[^\d.]', ''), 
                                             errors='coerce')
    
        # Clear existing database
        db.session.query(Car).delete()
        
        # Process and insert data
        for _, row in cars_df.iterrows():
            new_car = Car(
                price=row['price'],
                mileage=row['mileage'],
                year_of_registration=row['year_of_registration'],
                standard_make=row['standard_make'],
                standard_model=row['standard_model'])
            db.session.add(new_car)
        db.session.commit()

        # Preprocess data and train model
        processed_data, encoder, filtered_df = preprocess_data(cars_df, training=True)
        X = processed_data
        y = filtered_df['price'].values
        
        model = LinearRegression()
        model.fit(X, y)

        # Generate summary statistics
        summary = {
            'total_listings': int(len(filtered_df)),
            'average_price': float(filtered_df['price'].mean()),
            'min_price': float(filtered_df['price'].min()),
            'max_price': float(filtered_df['price'].max()),
            'average_mileage': float(filtered_df['mileage'].mean()),
            'most_common_makes': {k: int(v) for k, v in filtered_df['standard_make'].value_counts().head().to_dict().items()}}

        return jsonify(summary)
        
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict car price based on features
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            make:
              type: string
              description: Car manufacturer
              required: true
            model:
              type: string
              description: Car model
              required: true
            year:
              type: integer
              description: Year of registration
              required: true
            mileage:
              type: integer
              description: Current mileage of the car
              required: true
    responses:
      200:
        description: Predicted car price
    '''
    global model, encoder
    
    if model is None or encoder is None:
        return jsonify({"error": "Model not loaded. Call /reload first."}), 400
        
    # Validate required parameters
    required_params = ['make', 'model', 'year', 'mileage']
    if not all(param in request.json for param in required_params):
        return jsonify({"error": "Missing required parameters. Required: make, model, year, mileage"}), 400
        
    data = request.json
    
    try:
        # Create input DataFrame with parameters in specified order
        input_df = pd.DataFrame([{
            'standard_make': data['make'],
            'standard_model': data['model'],
            'year_of_registration': data['year'],
            'mileage': data['mileage'],
            'price': 0}])

        # Preprocess and predict
        processed_data, _, _ = preprocess_data(input_df, encoder=encoder, training=False)
        predicted_price = model.predict(processed_data)[0]
        return jsonify({"predicted_price": float(predicted_price)})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
