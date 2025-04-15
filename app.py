import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import urllib.request
import zipfile
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_features(data, features, imputer=None, scaler=None):
    """Prepare numeric features with optional imputation and scaling"""
    X = data[features].values
    if imputer:
        X = imputer.transform(X)
    if scaler:
        X = scaler.transform(X)
    return X

@st.cache_data
def load_data():
    """Load and cache data from GitHub"""
    try:
        url = 'https://github.com/DanaKChilds/AutoTrader/raw/refs/heads/master/AML_dataset.csv.zip'
        logger.info(f"Attempting to load data from {url}")
        with urllib.request.urlopen(url) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
                with zip_file.open(zip_file.namelist()[0]) as csv_file:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Successfully loaded data with {len(df)} rows")
                    return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return None
    
def clean_data(df):
    """Clean and filter data"""
    if df is None:
        return None
        
    numeric_cols = ['price', 'mileage', 'year_of_registration']
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)
    
    masks = {
        'price': (df['price'] >= 500) & (df['price'] <= 700000),
        'mileage': (df['mileage'] >= 0) & (df['mileage'] <= 400000),
        'year_of_registration': (df['year_of_registration'] >= 1980) & 
                              (df['year_of_registration'] <= 2025)
    }
    return df[np.all([masks[col] for col in numeric_cols], axis=0)]

def create_visualization(df, viz_type, selected_make):
    """Create visualization based on type"""
    try:
        viz_df = df if selected_make == "All" else df[df["standard_make"] == selected_make]
        
        if viz_type == "Price Distribution":
            return px.histogram(viz_df, x="price", 
                              title="Car Price Distribution",
                              labels={"price": "Price ($)", "count": "Number of Cars"})
        elif viz_type == "Price vs. Mileage":
            return px.scatter(viz_df, x="mileage", y="price",
                             title="Price vs. Mileage",
                             labels={"mileage": "Mileage", "price": "Price ($)"})
        else:
            avg_price = viz_df.groupby("standard_make")["price"].mean().reset_index()
            return px.bar(avg_price, x="standard_make", y="price",
                         title="Average Price by Make",
                         labels={"standard_make": "Make", "price": "Average Price ($)"})
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        st.error(f"Failed to create visualization: {str(e)}")
        return None

@st.cache_resource
def train_model(df):
    """Train and cache the price prediction model"""
    try:
        numeric_features = ['mileage', 'year_of_registration']
        X_numeric = df[numeric_features].values
        
        # Initialize and fit preprocessors
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        
        # Fit and transform numeric features
        X_numeric = imputer.fit_transform(X_numeric)
        X_numeric = scaler.fit_transform(X_numeric)
        
        # Prepare categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(df[['standard_make', 'standard_model']])
        
        # Combine features and train model
        X = np.column_stack((X_numeric, X_cat))
        model = Ridge(alpha=0.000001)
        model.fit(X, df['price'].values)
        
        return model, encoder, scaler, imputer
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        st.error(f"Failed to train model: {str(e)}")
        return None, None, None, None

def main():
    try:
        st.title("Auto Trader Price Predictor")
        
        # Add reload button in sidebar
        if st.sidebar.button("Reload Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Load and clean data
        with st.spinner('Loading data...'):
            df = load_data()
            
        if df is None:
            st.error("Failed to initialize application - data loading error")
            return
            
        with st.spinner('Processing data...'):
            df = clean_data(df)
            if df is None or df.empty:
                st.error("No valid data after cleaning")
                return

        # Display summary statistics
        st.sidebar.header("Summary Statistics")
        stats = {
            "Total Cars": f"{len(df):,}",
            "Average Price": f"${df['price'].mean():,.2f}",
            "Average Mileage": f"{df['mileage'].mean():,.0f}"
        }
        for label, value in stats.items():
            st.sidebar.write(f"{label}: {value}")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Price Prediction", "Data Analysis"])
        
        with tab1:
            st.header("Predict Car Price")
            
            # Input controls in two columns
            col1, col2 = st.columns(2)
            with col1:
                make = st.selectbox("Make", sorted(df["standard_make"].unique()), key="pred_make")
                year = st.number_input("Year", min_value=1980, max_value=2025, value=2020)
            with col2:
                models = sorted(df[df["standard_make"] == make]["standard_model"].unique())
                model = st.selectbox("Model", models, key="pred_model")
                mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=50000)
            
            if st.button("Predict"):
                model, encoder, scaler, imputer = train_model(df)
                if None in (model, encoder, scaler, imputer):
                    st.error("Failed to initialize model")
                    return
                    
                input_data = pd.DataFrame([{
                    'standard_make': make,
                    'standard_model': model,
                    'mileage': mileage,
                    'year_of_registration': year
                }])
                
                try:
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
                    predicted_price = model.predict(X)[0]
                    st.success(f"Predicted Price: ${predicted_price:,.2f}")
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    st.error(f"Failed to make prediction: {str(e)}")

        with tab2:
            viz_type = st.selectbox("Choose Visualization",
                                  ["Price Distribution", "Price vs. Mileage", "Average Price by Make"])
            makes = ["All"] + sorted(df["standard_make"].unique().tolist())
            selected_make = st.selectbox("Select Make", makes)
            
            fig = create_visualization(df, viz_type, selected_make)
            if fig is not None:
                st.plotly_chart(fig)
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()