import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        
        # Initialize encoders
        make_encoder = LabelEncoder()
        model_encoder = LabelEncoder()
        
        # Encode categorical features
        encoded_make = make_encoder.fit_transform(df['standard_make'])
        encoded_model = model_encoder.fit_transform(df['standard_model'])
        logger.info(f"Encoded {len(make_encoder.classes_)} makes and {len(model_encoder.classes_)} models")
        
        # Combine all features
        X = np.column_stack([
            df['mileage'].values,
            df['year_of_registration'].values,
            encoded_make,
            encoded_model
        ])
        logger.info(f"Combined feature matrix shape: {X.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train model
        ridge_model = Ridge(alpha=0.001)
        ridge_model.fit(X, df['price'].values)
        logger.info("Model training completed successfully")
        logger.info("-" * 50)
        
        return ridge_model, make_encoder, model_encoder, scaler
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
                ridge_model, make_encoder, model_encoder, scaler = train_model(df)
                
                if None in (ridge_model, make_encoder, model_encoder, scaler):
                    logger.error("Model initialization failed")
                    st.error("Failed to initialize model")
                    return
                
                try:
                    # Create input features
                    X = np.array([[
                        mileage,
                        year,
                        make_encoder.transform([make])[0],
                        model_encoder.transform([selected_model])[0]
                    ]])
                    logger.info(f"Created input array shape: {X.shape}")
                    
                    # Scale features
                    X = scaler.transform(X)
                    logger.info("Scaled input features")
                    
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