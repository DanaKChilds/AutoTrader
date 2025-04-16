# Auto Trader Car Price Predictor

A Streamlit web application that predicts used car prices based on make, model, year, and mileage using machine learning.

## Features

- Price prediction using Ridge Regression
- Interactive data visualizations
- User-friendly interface
- Data filtering and preprocessing

## Data Source

The data used for this project comes from the [Auto Trader Car Sale Adverts Dataset](https://www.kaggle.com/datasets/shayanshahid997/autotrader-at-car-sale-adverts-dataset), which provides detailed information about Auto Trader used vehicle listings.

The dataset includes important features such as:
- **Price**: The selling price of the car
- **Mileage**: The total mileage of the car
- **Year of Registration**: The year the car was registered
- **Make**: The manufacturer of the car
- **Model**: The model name of the car

## Prediction Process

The application uses a **Ridge Regression Model** to predict car prices based on various features:

1. **Data Preprocessing**: 
   - Cleaning of numerical values
   - Handling missing values using mean imputation
   - Scaling features using StandardScaler
   - Label encoding for categorical variables

2. **Model Training**: 
   - Ridge regression
   - Features include mileage, year, make, and model
   - Cached training for better performance

3. **Interactive Analysis**:
   - Price distribution visualization
   - Price vs. mileage scatter plots
   - Average price by make analysis

## Prerequisites

- Python 3.9+
- pip (Python package installer)
- Git (optional)

## Installation

1. **Clone or download the repository**:
```bash
git clone https://github.com/DanaKChilds/AutoTrader.git
cd AutoTrader
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. Select the "Price Prediction" tab to:
   - Choose a car make and model
   - Enter mileage and year
   - Get predicted price

2. Use the "Data Analysis" tab to:
   - View price distributions
   - Analyze price vs. mileage trends
   - Compare average prices by make

3. Use the reload button in the sidebar to refresh the data

## Project Structure

```
AutoTrader/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
```

## Dependencies

- streamlit
- pandas
- plotly
- scikit-learn
- numpy
