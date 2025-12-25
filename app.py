import streamlit as st
import pandas as pd
import joblib

raw_data = pd.read_csv("C:\\Users\\Sudharsaa\\Documents\\221501149\\HeyRam\\HeyRam_Task\\data\\raw_kaggle_data.csv")
model = joblib.load("C:\\Users\\Sudharsaa\\Documents\\221501149\\HeyRam\\HeyRam_Task\\model\\catboost_model_v2.pkl")
data = pd.read_csv("C:\\Users\\Sudharsaa\\Documents\\221501149\\HeyRam\\HeyRam_Task\\data\\cleaned_data.csv")

FEATURES = model.feature_names_

REAL_COORDS = {
    "Adyar": (13.0067, 80.2570),
    "Velachery": (12.9784, 80.2210),
    "Tambaram": (12.9249, 80.1275),
    "East Tambaram": (12.9249, 80.1275),
    "West Tambaram": (12.9249, 80.1175),
    "Chromepet": (12.9525, 80.1460),
    "Pallavaram": (12.9675, 80.1490),
    "Medavakkam": (12.9180, 80.1920),
    "Perungudi": (12.9716, 80.2446),
    "Thoraipakkam": (12.9395, 80.2350),
    "Sholinganallur": (12.8990, 80.2270),
    "T Nagar": (13.0418, 80.2341),
    "Anna Nagar": (13.0850, 80.2101),
    "Ambattur": (13.1143, 80.1480),
    "Avadi": (13.1145, 80.1090),
    "Porur": (13.0382, 80.1565),
    "OMR": (12.9000, 80.2300),
    "ECR": (12.9150, 80.2530)
}

st.set_page_config(page_title="FutureNest", layout="wide")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;color:#0B3C5D;'>🏠 FutureNest</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>Chennai Real Estate Price Prediction System</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Enter Property Details")

year = st.sidebar.number_input("Purchase Year", 2024, 2100, 2026, 1)
location = st.sidebar.selectbox("Location", sorted(data["location"].astype(str).unique()))
bhk = st.sidebar.number_input("BHK", 1, 10, 2, 1)
area = st.sidebar.number_input("Total Area (sqft)", 300.0, 10000.0, 1200.0)

predict_btn = st.sidebar.button("Predict Price")

st.sidebar.markdown("---")
st.sidebar.markdown("### ML Process Explorer")

btn_about = st.sidebar.button("About Dataset")
btn_eda = st.sidebar.button("EDA Analysis")
btn_clean = st.sidebar.button(" Data Cleaning")
btn_model = st.sidebar.button("Model Building")
btn_hide = st.sidebar.button("Perdition Panel")

if btn_about:
    st.session_state.process = "About Dataset"
elif btn_eda:
    st.session_state.process = "EDA"
elif btn_clean:
    st.session_state.process = "Data Cleaning"
elif btn_model:
    st.session_state.process = "Model Building"
elif btn_hide:
    st.session_state.process = "None"

process_view = st.session_state.get("process", "None")

# ---------------- PREDICTION ----------------
if predict_btn:
    user_input = {"location": location, "bhk": int(bhk), "area": float(area), "year": int(year)}
    input_df = pd.DataFrame([[user_input[f] for f in FEATURES]], columns=FEATURES)

    base_price = model.predict(input_df)[0] * 100000
    growth_rate = 0.08
    prediction = base_price * ((1 + growth_rate) ** (year - 2024))

    st.markdown(f"""
    <div style='background:#D4EFDF;padding:30px;border-radius:15px;text-align:center;'>
    <h2 style='color:black;'>Estimated House Price</h2>
    <h1 style='color:#117A65;'>₹ {prediction:,.0f}</h1>
    </div>
    """, unsafe_allow_html=True)

    actual_df = data[data["location"] == location]
    actual_year_avg = actual_df.groupby("year")["price"].mean().sort_index()
    actual_years = actual_year_avg.index.tolist()
    actual_prices = (actual_year_avg.values * 100000).tolist()

    last_actual_year = int(actual_years[-1])
    last_actual_price = actual_prices[-1]

    future_years = list(range(last_actual_year, int(year) + 1))
    future_prices = [base_price * ((1 + growth_rate) ** (y - 2024)) for y in future_years]
    future_prices[0] = last_actual_price

    combined_df = pd.DataFrame({
        "Actual Market Price": pd.Series(actual_prices, index=actual_years),
        "Predicted Future Price": pd.Series(future_prices, index=future_years)
    })


    st.markdown("### Property Location Map")
    lat, lon = REAL_COORDS.get(location, (13.05, 80.25))
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

# ---------------- PROCESS PANELS ----------------
if process_view == "About Dataset":
    st.markdown("## About the Dataset")

    st.markdown("""
    • This dataset contains house prices in Chennai.  
    • It is used to predict house prices using Machine Learning.

    • The dataset has **8 columns**.  
    • **Price** is the target column.  
    • The other **7 columns** are features that affect house prices.

    • Data was collected from **makaan.com (Oct 2021)**.

    • This dataset helps people estimate how much they need to spend to buy a house in Chennai  
    based on location, builder and property features.
    """)
    st.markdown("### Dataset Snapshot")
    st.dataframe(raw_data.head(5))

elif process_view == "EDA":
    st.markdown("##  Exploratory Data Analysis")
    st.dataframe(raw_data.head(50))
    st.bar_chart(raw_data["price"].value_counts().head(20))
    st.line_chart(raw_data.groupby("bhk")["price"].mean())
    st.scatter_chart(raw_data, x="area", y="price")

elif process_view == "Data Cleaning":
    
    st.markdown("### Cleaned Data Sample")
    st.dataframe(data.sample(50)) 


    st.markdown("## Data Cleaning & Preprocessing")

    cleaning_steps = pd.DataFrame({
        "Step": [
            "Missing Value Handling",
            "Duplicate Removal",
            "Text Normalization",
            "Outlier Treatment",
            "Feature Engineering",
            "Final Dataset Preparation"
        ],
        "Description": [
            "Filled missing BHK, Area, and Price using median values. Removed rows with critical missing locations.",
            "Removed duplicate property listings to avoid bias.",
            "Standardized location and builder names. Removed extra spaces and symbols.",
            "Removed extremely high/low price and abnormal area outliers.",
            "Converted mixed sqft text into numeric values. Extracted BHK and year features.",
            "Converted categorical features into model-friendly format and finalized clean dataset."
        ]
    })

    st.table(cleaning_steps)

    
elif process_view == "Model Building":
    st.markdown("##  Model Benchmarking")
    st.table(pd.DataFrame({
        "Model": ["Linear Regression", "LightGBM", "CatBoost"],
        "R² (%)": [85.69, 84.17, 89.68],
        "MAE": [9.68, 10.49, 8.29],
        "RMSE": [15.38, 16.17, 13.05],
        "Status": ["Baseline", "Better", "✅ Selected"]
    }))
    
    import altair as alt

    st.markdown("### Model Performance Comparison")

    models = ["Linear Regression", "LightGBM", "CatBoost"]

    r2_vals   = [85.69, 84.17, 89.68]
    mae_vals  = [9.68, 10.49, 8.29]
    rmse_vals = [15.38, 16.17, 13.05]

    best_r2   = max(r2_vals)
    best_mae  = min(mae_vals)
    best_rmse = min(rmse_vals)

    df_perf = pd.DataFrame({
        "Model": models,
        "R2": r2_vals,
        "MAE": mae_vals,
        "RMSE": rmse_vals
    })

    c1, c2, c3 = st.columns(3)

    # ---------- R2 ----------
    with c1:
        st.markdown("#### 🟢 R² Score (%)")
        chart = alt.Chart(df_perf).mark_bar().encode(
            x="Model",
            y="R2",
            color=alt.condition(
                alt.datum.R2 == best_r2,
                alt.value("#2ECC71"),     # Best = green
                alt.value("#3498DB")      # Others = blue
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # ---------- MAE ----------
    with c2:
        st.markdown("#### 🟠 MAE (Lower is Better)")
        chart = alt.Chart(df_perf).mark_bar().encode(
            x="Model",
            y="MAE",
            color=alt.condition(
                alt.datum.MAE == best_mae,
                alt.value("#F39C12"),     # Best = orange
                alt.value("#3498DB")
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # ---------- RMSE ----------
    with c3:
        st.markdown("#### 🔴 RMSE (Lower is Better)")
        chart = alt.Chart(df_perf).mark_bar().encode(
            x="Model",
            y="RMSE",
            color=alt.condition(
                alt.datum.RMSE == best_rmse,
                alt.value("#E74C3C"),     # Best = red
                alt.value("#3498DB")
            )
        )
        st.altair_chart(chart, use_container_width=True)


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed by Sudharsan S</p>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;font-size:13px;color:gray;'>
⚠ Disclaimer: FutureNest provides estimated prices based on historical and simulated growth data.
Predicted values are for informational purposes only.
</p>
""", unsafe_allow_html=True)
