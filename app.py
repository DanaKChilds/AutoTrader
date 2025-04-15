import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import urllib.request
import zipfile
import io

@st.cache_data(ttl=3600)
def load_and_clean_data():
    """Load and clean data from GitHub"""
    url = 'https://github.com/DanaKChilds/AutoTrader/raw/refs/heads/master/AML_dataset.csv.zip'
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
            with zip_file.open(zip_file.namelist()[0]) as csv_file:
                df = pd.read_csv(csv_file)
    
    # Clean data
    numeric_cols = ['price', 'mileage', 'year_of_registration']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Apply filters
    return df[
        (df['price'].between(500, 700000)) &
        (df['mileage'].between(0, 400000)) &
        (df['year_of_registration'].between(1980, 2025))
    ].dropna(subset=numeric_cols)

@st.cache_resource
def train_model(df):
    """Train price prediction model"""
    # Encode categorical features
    encoders = {
        'make': LabelEncoder().fit(df['standard_make']),
        'model': LabelEncoder().fit(df['standard_model'])
    }
    
    # Prepare features
    X = np.column_stack([
        df[['mileage', 'year_of_registration']].values,
        encoders['make'].transform(df['standard_make']),
        encoders['model'].transform(df['standard_model'])
    ])
    
    # Scale and train
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=0.001).fit(scaler.transform(X), df['price'])
    
    return model, encoders, scaler

def create_visualization(df, viz_type, selected_make):
    """Create visualization"""
    df_viz = df if selected_make == "All" else df[df["standard_make"] == selected_make]
    
    if viz_type == "Price Distribution":
        plot = px.histogram(df_viz, x="price", title="Car Price Distribution")
    elif viz_type == "Price vs. Mileage":
        plot = px.scatter(df_viz, x="mileage", y="price", title="Price vs. Mileage")
    else:
        avg_price = df_viz.groupby("standard_make")["price"].mean().reset_index()
        plot = px.bar(avg_price, x="standard_make", y="price", title="Average Price by Make")
    
    plot.update_layout(
        xaxis_title="Price ($)" if viz_type == "Price Distribution" else "Mileage",
        yaxis_title="Count" if viz_type == "Price Distribution" else "Price ($)"
    )
    return plot

def main():
    st.title("Auto Trader Price Predictor")
    
    # Load data
    with st.spinner('Loading and processing data...'):
        df = load_and_clean_data()
    
    # Sidebar stats
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
    
    # Prediction tab
    with tab1:
        st.header("Predict Car Price")
        col1, col2 = st.columns(2)
        
        with col1:
            make = st.selectbox("Make", sorted(df["standard_make"].unique()))
            year = st.number_input("Year", 1980, 2025, 2020)
        
        with col2:
            model = st.selectbox("Model", sorted(df[df["standard_make"] == make]["standard_model"].unique()))
            mileage = st.number_input("Mileage", 0, 500000, 50000)
        
        if st.button("Predict"):
            ridge_model, encoders, scaler = train_model(df)
            X = np.array([[
                mileage,
                year,
                encoders['make'].transform([make])[0],
                encoders['model'].transform([model])[0]
            ]])
            predicted_price = ridge_model.predict(scaler.transform(X))[0]
            st.success(f"Predicted Price: ${predicted_price:,.2f}")
    
    # Analysis tab
    with tab2:
        viz_type = st.selectbox("Visualization Type", 
                              ["Price Distribution", "Price vs. Mileage", "Average Price by Make"])
        makes = ["All"] + sorted(df["standard_make"].unique().tolist())
        selected_make = st.selectbox("Select Make", makes)
        st.plotly_chart(create_visualization(df, viz_type, selected_make))

if __name__ == "__main__":
    main()