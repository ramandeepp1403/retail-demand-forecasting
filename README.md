# 📈 Retail Demand Forecasting

An end-to-end Machine Learning project that predicts future retail sales for a selected **Store**, **Item**, and **Forecast Date** using historical sales patterns, feature engineering, and Linear Regression.

## 🌐 Live Demo

**🚀 Streamlit App:**
https://retail-demand-forecasting-dhfnrskyxcwsozxamt3rm8.streamlit.app/

---

## 📌 Project Overview

Retail businesses need accurate demand forecasting to avoid running out of stock or overstocking products.

This project predicts the expected daily sales of a selected item in a selected store by learning from historical sales data between **2013 and 2017** and forecasting demand for **January 2018**.

Users simply select:

* 🏪 Store
* 📦 Item
* 📅 Forecast Date

The application then estimates the expected number of units that will be sold.

---

## 🎯 Problem Statement

Retailers often struggle with:

* Stock shortages
* Overstocking
* Poor inventory planning
* Inaccurate demand estimation

This project uses historical sales trends and machine learning to provide demand forecasts that can support inventory planning and business decision-making.

---

## 📂 Dataset

**Kaggle:** Store Item Demand Forecasting Challenge

The dataset contains:

* 5 Years of Daily Sales Data
* 10 Stores
* 50 Different Items
* Sales Records from January 2013 to December 2017
## Dataset Columns

| Column | Description |
|---------|-------------|
| **date** | Date of the sales transaction |
| **store** | Store identifier (1–10) |
| **item** | Item identifier (1–50) |
| **sales** | Daily units sold (Target Variable)

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Preprocessing

* Date parsing
* Missing value checking
* Feature creation
* Train-Test split
* Feature scaling

---

### 2️⃣ Feature Engineering

Calendar Features

* Day of Week
* Month
* Year
* Week Number
* Quarter
* Day of Year
* Weekend Indicator

Historical Features

* Lag 1
* Lag 2
* Lag 3
* Lag 7
* Lag 14
* Lag 21
* Lag 30

Rolling Statistics

* Rolling Mean (7, 14, 30 days)
* Rolling Standard Deviation (7, 14, 30 days)

---

### 3️⃣ Models Compared

| Model                 |      MAE |     RMSE |  R² Score |
| --------------------- | -------: | -------: | --------: |
| **Linear Regression** | **5.98** | **7.74** | **0.907** |
| Decision Tree         |     8.02 |    10.49 |     0.829 |
| CNN                   |     9.80 |    13.05 |     0.736 |

✅ **Linear Regression achieved the best overall performance and was selected for deployment.**

---

## 📊 Model Performance

**Mean Absolute Error (MAE):** 5.98

**Root Mean Squared Error (RMSE):** 7.74

**R² Score:** 0.907

These results indicate that the model explains approximately **91% of the variance** in sales, demonstrating strong predictive performance on the test dataset.

---

## 💻 Streamlit Dashboard Features

* Interactive Store Selection
* Item Selection
* Forecast Date Selection
* Sales Prediction
* Historical Sales Trend
* Model Performance Metrics
* Model Comparison Table
* Engineered Features Viewer

---

## 📸 Application Preview

<img width="1907" height="1008" alt="Screenshot 2026-06-27 000632" src="https://github.com/user-attachments/assets/735ec527-3b7c-4fb5-87c8-f07f396a1492" />



---

## 🚀 How to Run Locally

Clone the repository

```bash
git clone https://github.com/ramandeepp1403/retail-demand-forecasting.git
```

Move into the project folder

```bash
cd retail-demand-forecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Retail-Demand-Forecasting
│
├── app.py
├── train.csv
├── linear_regression_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── Retail_Forecasting_End_to_End.ipynb
```

---

## 🔮 Future Improvements

* Random Forest and XGBoost comparison
* LSTM-based time series forecasting
* Multi-day sales forecasting
* Confidence intervals for predictions
* REST API deployment using FastAPI
* Docker containerization
* Cloud deployment with CI/CD

---

## 👨‍💻 Author

**Ramandeep Singh**

* GitHub: https://github.com/ramandeepp1403
* LinkedIn: *https://www.linkedin.com/in/ramandeep-pandi/*

---

## ⭐ Support

If you found this project useful, consider giving the repository a **⭐ Star**.
It helps others discover the project and motivates future improvements.
