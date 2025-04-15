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

def format_currency(value):
    """Format value as GBP currency"""
    return f"£{value:,.2f}"

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
    url = 'https://github.com/DanaKChilds/AutoTrader/raw/refs/heads/master/AML_dataset.csv.zip'
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
            with zip_file.open(zip_file.namelist()[0]) as csv_file:
                return pd.read_csv(csv_file)

def clean_data(df):
    """Clean and filter data"""
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
    viz_df = df if selected_make == "All" else df[df["standard_make"] == selected_make]
    
    if viz_type == "Price Distribution":
        return px.histogram(viz_df, x="price", 
                          title="Car Price Distribution",
                          labels={"price": "Price (£)", "count": "Number of Cars"})
    elif viz_type == "Price vs. Mileage":
        return px.scatter(viz_df, x="mileage", y="price",
                         title="Price vs. Mileage",
                         labels={"mileage": "Mileage", "price": "Price (£)"})
    else:
        avg_price = viz_df.groupby("standard_make")["price"].mean().reset_index()
        return px.bar(avg_price, x="standard_make", y="price",
                     title="Average Price by Make",
                     labels={"standard_make": "Make", "price": "Average Price (£)"})

@st.cache_resource
def train_model(df):
    """Train and cache the price prediction model"""
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

def main():
    st.title("Auto Trader Price Predictor")
    
    # Add reload button in sidebar
    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Display summary statistics
    st.sidebar.header("Summary Statistics")
    stats = {
        "Total Cars": f"{len(df):,}",
        "Average Price": format_currency(df['price'].mean()),
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
            input_data = pd.DataFrame([{
                'standard_make': make,
                'standard_model': model,
                'mileage': mileage,
                'year_of_registration': year
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
            predicted_price = model.predict(X)[0]
            st.success(f"Predicted Price: {format_currency(predicted_price)}")

    with tab2:
        viz_type = st.selectbox("Choose Visualization",
                              ["Price Distribution", "Price vs. Mileage", "Average Price by Make"])
        makes = ["All"] + sorted(df["standard_make"].unique().tolist())
        selected_make = st.selectbox("Select Make", makes)
        
        fig = create_visualization(df, viz_type, selected_make)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()