import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="House Price AI", layout="wide")

# -------------------------------
# CUSTOM CSS (MODERN UI)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

/* FIX: make text visible */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    color: black;
}

.metric-card h2 {
    color: black;
}

.metric-card h3 {
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
model = joblib.load("models/house_price_model.pkl")
df = pd.read_csv("data/processed/cleaned_data.csv")

# -------------------------------
# TITLE
# -------------------------------
st.title("🏠 AI-Powered House Price Prediction")
st.markdown("### Smart Real Estate Analytics Dashboard")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("📌 Enter Property Details")

area = st.sidebar.number_input("Area (sqft)", 300, 10000, 1200)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
balcony = st.sidebar.slider("Balconies", 0, 5, 1)
location = st.sidebar.text_input("Location", "Whitefield")

# Derived features
total_rooms = bedrooms + bathrooms
area_type = st.sidebar.selectbox(
    "Area Type",
    ["Super built-up  Area", "Built-up  Area", "Plot  Area"]
)
# -------------------------------
# PREDICTION
# -------------------------------
# -------------------------------
# CREATE INPUT DATA (MOVE HERE)
# -------------------------------
input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "balcony": [balcony],
    "location": [location],
    "area_type": [area_type],   # important
    "total_rooms": [bedrooms + bathrooms],
    "price_per_sqft": [0],
    "is_luxury": [0]
})

# -------------------------------
# PREDICT BUTTON
# -------------------------------
predict_btn = st.sidebar.button("🚀 Predict Price")

if predict_btn:
    prediction = model.predict(input_data)[0]

    # calculate derived values
    price_per_sqft = prediction / area
    luxury = "Luxury 🏆" if prediction > df["price"].median() else "Budget 💰"

    # store everything
    st.session_state["prediction"] = prediction
    st.session_state["price_per_sqft"] = price_per_sqft
    st.session_state["luxury"] = luxury

# -------------------------------
# KEEP RESULT VISIBLE
# -------------------------------
if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    price_per_sqft = st.session_state["price_per_sqft"]
    luxury = st.session_state["luxury"]
    # Luxury classification
    luxury = "Luxury 🏆" if prediction > df["price"].median() else "Budget 💰"

    # -------------------------------
    # KPI CARDS
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h3>💰 Predicted Price</h3>
        <h2>₹ {prediction:,.2f} L</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h3>📐 Price / Sqft</h3>
        <h2>₹ {price_per_sqft:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h3>🏷️ Category</h3>
        <h2>{luxury}</h2>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Insights", "🤖 Model"])

# -------------------------------
# TAB 1: DATA
# -------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.sample(10))

    st.subheader("Summary Statistics")
    st.write(df.describe())

# -------------------------------
# TAB 2: VISUALS
# -------------------------------
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["price"], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Area vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df["area"], y=df["price"], ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ================================
    # 🔥 ADD YOUR NEW CHART HERE
    # ================================

    st.subheader("🏆 Price Comparison")

    if "prediction" in st.session_state:
        predicted_price = st.session_state["prediction"]

        avg_price = df["price"].mean()
        max_price = df["price"].max()

        comparison_df = pd.DataFrame({
            "Category": ["Predicted", "Average", "Maximum"],
            "Price": [predicted_price, avg_price, max_price]
        })

        fig, ax = plt.subplots()

        colors = ["green", "blue", "red"]
        ax.bar(comparison_df["Category"], comparison_df["Price"], color=colors)

        for i, v in enumerate(comparison_df["Price"]):
            ax.text(i, v + 5, f"{v:.1f}", ha='center')

        ax.set_title("Price Comparison Analysis")
        ax.set_ylabel("Price (Lakhs)")

        st.pyplot(fig)

    else:
        st.info("Click 'Predict Price' to see comparison chart")
# -------------------------------
# TAB 3: MODEL INFO
# -------------------------------
with tab3:
    st.subheader("Model Overview")

    st.markdown("""
    ### 🚀 Models Used:
    - Linear Regression
    - Random Forest (Tuned)
    - XGBoost

    ### 🎯 Key Features:
    - Area (sqft)
    - Bedrooms & Bathrooms
    - Location
    - Engineered Features:
        - Price per sqft
        - Total rooms
        - Luxury classification

    ### 📊 Metrics:
    - MAE
    - RMSE
    - R² Score

    ### 💡 Insight:
    Random Forest & XGBoost performed best due to non-linear patterns in housing data.
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built with ❤️ | Machine Learning Project | Streamlit Dashboard")