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

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def prepare_features(data, features, imputer=None, scaler=None):
    """Prepare numeric features with optional imputation and scaling"""
    try:
        logger.info(f"Preparing features: {features}")
        X = data[features].values
        if imputer:
            X = imputer.transform(X)
        if scaler:
            X = scaler.transform(X)
        logger.info(f"Features prepared successfully. Shape: {X.shape}")
        return X
    except Exception as e:
        logger.error(f"Feature preparation error: {str(e)}", exc_info=True)
        raise

@st.cache_data(ttl=3600)
def load_data():
    """Load and cache data from GitHub"""
    try:
        url = 'https://github.com/DanaKChilds/AutoTrader/raw/refs/heads/master/AML_dataset.csv.zip'
        logger.info(f"Attempting to load data from {url}")
        with urllib.request.urlopen(url) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
                with zip_file.open(zip_file.namelist()[0]) as csv_file:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Successfully loaded data with {len(df):,} rows")
                    return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}", exc_info=True)
        st.error(f"Failed to load data: {str(e)}")
        return None

def clean_data(df):
    """Clean and filter data"""
    try:
        logger.info("Starting data cleaning")
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
        cleaned_df = df[np.all([masks[col] for col in numeric_cols], axis=0)]
        logger.info(f"Data cleaning complete. Rows remaining: {len(cleaned_df):,}")
        return cleaned_df
    except Exception as e:
        logger.error(f"Data cleaning error: {str(e)}", exc_info=True)
        raise

def create_visualization(df, viz_type, selected_make):
    """Create visualization based on type"""
    try:
        logger.info(f"Creating {viz_type} visualization for {selected_make}")
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
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        st.error(f"Failed to create visualization: {str(e)}")
        return None

@st.cache_resource
def train_model(df):
    """Train and cache the price prediction model"""
    try:
        logger.info("-" * 50)
        logger.info("Starting model training")
        numeric_features = ['mileage', 'year_of_registration']
        X_numeric = df[numeric_features].values
        logger.info(f"Numeric features shape: {X_numeric.shape}")
        
        # Initialize and fit preprocessors
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        
        # Fit and transform numeric features
        X_numeric = imputer.fit_transform(X_numeric)
        X_numeric = scaler.fit_transform(X_numeric)
        logger.info("Numeric features processed")
        
        # Prepare categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(df[['standard_make', 'standard_model']])
        logger.info(f"Categorical features shape: {X_cat.shape}")
        
        # Combine features and train model
        X = np.column_stack((X_numeric, X_cat))
        ridge_model = Ridge(alpha=0.000001)
        ridge_model.fit(X, df['price'].values)
        logger.info("Model training completed successfully")
        logger.info("-" * 50)
        
        return ridge_model, encoder, scaler, imputer
    except Exception as e:
        logger.error(f"Model training error: {str(e)}", exc_info=True)
        st.error(f"Failed to train model: {str(e)}")
        return None, None, None, None

def main():
    try:
        logger.info("Starting application")
        st.title("Auto Trader Price Predictor")
        
        # Add reload button in sidebar
        if st.sidebar.button("Reload Data"):
            logger.info("Data reload requested")
            st.cache_data.clear()
            st.rerun()
        
        # Load and clean data
        with st.spinner('Loading data...'):
            logger.info("Loading data")
            df = load_data()
            
        if df is None:
            logger.error("Data loading failed")
            st.error("Failed to initialize application - data loading error")
            return
            
        with st.spinner('Processing data...'):
            df = clean_data(df)
            if df is None or df.empty:
                logger.error("Data cleaning resulted in empty dataset")
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
                selected_model = st.selectbox("Model", models, key="pred_model")
                mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=50000)
            
            if st.button("Predict"):
                logger.info("-" * 50)
                logger.info("Starting prediction process")
                ridge_model, encoder, scaler, imputer = train_model(df)
                if None in (ridge_model, encoder, scaler, imputer):
                    logger.error("Model initialization failed")
                    st.error("Failed to initialize model")
                    return
                
                logger.info(f"Input data - Make: {make}, Model: {selected_model}, Year: {year}, Mileage: {mileage}")
                input_data = pd.DataFrame([{
                    'standard_make': make,
                    'standard_model': selected_model,
                    'mileage': mileage,
                    'year_of_registration': year
                }])
                
                try:
                    # Process features
                    logger.info("Processing input features")
                    X_numeric = prepare_features(
                        input_data, 
                        ['mileage', 'year_of_registration'], 
                        imputer, 
                        scaler
                    )
                    logger.info(f"Numeric features processed: {X_numeric.shape}")
                    
                    X_cat = encoder.transform(input_data[['standard_make', 'standard_model']])
                    logger.info(f"Categorical features processed: {X_cat.shape}")
                    
                    X = np.column_stack((X_numeric, X_cat))
                    logger.info(f"Final feature matrix shape: {X.shape}")
                    
                    # Make prediction
                    predicted_price = ridge_model.predict(X)[0]
                    logger.info(f"Prediction successful: ${predicted_price:,.2f}")
                    st.success(f"Predicted Price: ${predicted_price:,.2f}")
                    logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)
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
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()