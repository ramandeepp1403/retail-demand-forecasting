import streamlit as st
import pandas as pd
import joblib
#page config
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide"
)
#final model 
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("train.csv", parse_dates=["date"])
#sidebar
with st.sidebar:
    st.title("📊 Project Information")

    st.markdown("### Model")
    st.success("Linear Regression")

    st.markdown("### Dataset")
    st.write("Store Item Demand Forecasting")

    st.markdown("### Training Period")
    st.write("2013 - 2017")

    st.markdown("### Forecast Period")
    st.write("January 2018")

    st.markdown("### Technologies")
    st.write("""
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
""")

st.title("📈 Retail Demand Forecasting")
st.markdown("""
Forecast the expected **daily sales** for a selected **Store** and **Item**
using historical demand patterns and machine learning.
""")

st.divider()

#user i/p
store = st.selectbox(
    "🏪 Store",
    sorted(df["store"].unique())
)

item = st.selectbox(
    "📦 Item",
    sorted(df["item"].unique())
)

forecast_date = st.date_input(
    "📅 Forecast Date",
    value=pd.Timestamp("2018-01-01"),
    min_value=pd.Timestamp("2018-01-01"),
    max_value=pd.Timestamp("2018-01-31")
)

#pred begins here
if st.button("🔮 Predict Sales", use_container_width=True):

    history = df[
        (df["store"] == store) &
        (df["item"] == item) &
        (df["date"] < pd.to_datetime(forecast_date))
    ].sort_values("date")

    if len(history) < 30:
        st.error("Not enough historical data to generate prediction.")
        st.stop()

    #calendar features
    date = pd.to_datetime(forecast_date)

    day_of_week = date.dayofweek
    month = date.month
    year = date.year
    day_of_year = date.dayofyear
    week = int(date.isocalendar().week)
    quarter = date.quarter
    is_weekend = 1 if day_of_week >= 5 else 0
#features used for pred
    sales = history["sales"]

    lag_1 = sales.iloc[-1]
    lag_2 = sales.iloc[-2]
    lag_3 = sales.iloc[-3]
    lag_7 = sales.iloc[-7]
    lag_14 = sales.iloc[-14]
    lag_21 = sales.iloc[-21]
    lag_30 = sales.iloc[-30]

    roll_mean_7 = sales.tail(7).mean()
    roll_std_7 = sales.tail(7).std()

    roll_mean_14 = sales.tail(14).mean()
    roll_std_14 = sales.tail(14).std()

    roll_mean_30 = sales.tail(30).mean()
    roll_std_30 = sales.tail(30).std()

# input giving 
    input_df = pd.DataFrame({

        "store": [store],
        "item": [item],
        "day_of_week": [day_of_week],
        "month": [month],
        "year": [year],
        "day_of_year": [day_of_year],
        "week": [week],
        "quarter": [quarter],
        "is_weekend": [is_weekend],

        "lag_1": [lag_1],
        "lag_2": [lag_2],
        "lag_3": [lag_3],
        "lag_7": [lag_7],
        "lag_14": [lag_14],
        "lag_21": [lag_21],
        "lag_30": [lag_30],

        "roll_mean_7": [roll_mean_7],
        "roll_std_7": [roll_std_7],

        "roll_mean_14": [roll_mean_14],
        "roll_std_14": [roll_std_14],

        "roll_mean_30": [roll_mean_30],
        "roll_std_30": [roll_std_30]

    })

    input_scaled = scaler.transform(input_df)

# predict
    prediction = model.predict(input_scaled)

    st.success(
        f"📦 Estimated Sales on {forecast_date.strftime('%d %b %Y')}: "
        f"**{round(prediction[0])} Units**"
    )

    st.info(
        "This forecast is generated using historical sales trends, lag features, "
        "rolling statistics, and calendar-based features learned from sales data "
        "between 2013 and 2017."
    )

    st.markdown("---")

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("MAE", "5.98")

    with c2:
        st.metric("RMSE", "7.74")

    with c3:
        st.metric("R² Score", "0.907")
 #summary
    st.markdown("---")

    st.subheader("📋 Forecast Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Store", store)
        st.metric("Item", item)

    with col2:
        st.metric(
            "Forecast Date",
            forecast_date.strftime("%d %b %Y")
        )
        st.metric(
            "Estimated Sales",
            f"{round(prediction[0])} Units"
        )

 #trend
    st.markdown("---")

    st.subheader("📈 Historical Sales Trend (Last 30 Days)")

    chart = history.tail(30).set_index("date")["sales"]

    st.line_chart(
    chart,
    height=350,
    use_container_width=True
)

#model comparsion
    st.markdown("---")

    st.subheader("🏆 Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "CNN"
        ],
        "MAE": [
            5.98,
            8.02,
            9.80
        ],
        "RMSE": [
            7.74,
            10.49,
            13.05
        ],
        "R² Score": [
            0.907,
            0.829,
            0.736
        ]
    })

    st.dataframe(comparison, use_container_width=True)

   
    with st.expander("🔍 View Features Used for Prediction"):
        st.dataframe(input_df, use_container_width=True)


st.markdown("---")

st.caption(
"""
Developed by **Ramandeep Singh**

Model: **Linear Regression**

Dataset: **Kaggle Store Item Demand Forecasting**

Built using **Python | Pandas | Scikit-learn | Streamlit**
"""
)