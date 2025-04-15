import pytest
import pandas as pd
import numpy as np
from app import load_data, clean_data, train_model, prepare_features

# Test data fixtures
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'price': [10000, 20000, 30000],
        'mileage': [50000, 60000, 70000],
        'year_of_registration': [2015, 2016, 2017],
        'standard_make': ['Toyota', 'Honda', 'Ford'],
        'standard_model': ['Camry', 'Civic', 'Focus']
    })

def test_clean_data(sample_data):
    """Test data cleaning function"""
    cleaned_df = clean_data(sample_data)
    assert not cleaned_df.isnull().any().any()
    assert (cleaned_df['price'] >= 500).all()
    assert (cleaned_df['price'] <= 700000).all()
    assert (cleaned_df['mileage'] >= 0).all()
    assert (cleaned_df['mileage'] <= 400000).all()

def test_prepare_features():
    """Test feature preparation"""
    test_data = pd.DataFrame({
        'mileage': [50000, 60000],
        'year_of_registration': [2015, 2016]
    })
    features = ['mileage', 'year_of_registration']
    
    X = prepare_features(test_data, features)
    assert X.shape == (2, 2)
    assert isinstance(X, np.ndarray)

def test_train_model(sample_data):
    """Test model training"""
    model, encoder, scaler, imputer = train_model(sample_data)
    
    # Test model exists and has expected attributes
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')
    
    # Test encoder configuration
    assert encoder.sparse_output == False
    assert encoder.handle_unknown == 'ignore'
    
    # Test preprocessors
    assert hasattr(scaler, 'transform')
    assert hasattr(imputer, 'transform')

def test_prediction_pipeline(sample_data):
    """Test complete prediction pipeline"""
    # Train model
    model, encoder, scaler, imputer = train_model(sample_data)
    
    # Create test input
    input_data = pd.DataFrame([{
        'standard_make': 'Toyota',
        'standard_model': 'Camry',
        'mileage': 50000,
        'year_of_registration': 2015
    }])
    
    # Process features
    X_numeric = prepare_features(
        input_data,
        ['mileage', 'year_of_registration'],
        imputer,
        scaler
    )
    X_cat = encoder.transform(input_data[['standard_make', 'standard_model']])
    X = np.column_stack((X_numeric, X_cat))
    
    # Make prediction
    prediction = model.predict(X)[0]
    assert isinstance(prediction, (float, np.float64))
    assert prediction > 0
