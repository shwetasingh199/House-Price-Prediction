# 🏠 House Price Prediction using Machine Learning

![App Banner](images/app_ui.png)

---

## 📌 Project Overview

This project builds an **end-to-end Machine Learning system** to predict house prices based on property features like area, location, number of bedrooms, and more.

It includes:

* Data Cleaning & Feature Engineering
* Multiple Regression Models
* Hyperparameter Tuning
* Interactive **Streamlit Web App**
* Real-time Price Prediction + Analytics Dashboard

---

## 🎯 Problem Statement

Accurately estimating house prices is crucial for:

* Buyers → making smart decisions
* Sellers → pricing correctly
* Real estate companies → market analysis
* Banks → loan valuation

This project solves that using **Regression Models + Data Science**.

---

## 🚀 Features

✔ Cleaned real-world housing dataset
✔ Feature engineering (price per sqft, total rooms, luxury flag)
✔ Multiple ML models:

* Linear Regression
* Random Forest (Tuned)
* XGBoost

✔ Model evaluation using:

* MAE
* RMSE
* R² Score

✔ Interactive dashboard with:

* Real-time prediction
* Price comparison chart
* Data visualizations

---

## 🧠 Machine Learning Workflow

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Evaluation → Deployment (Streamlit)
```

---

## 📊 Tech Stack

| Category      | Tools                 |
| ------------- | --------------------- |
| Language      | Python                |
| Data          | Pandas, NumPy         |
| Visualization | Matplotlib, Seaborn   |
| ML Models     | Scikit-learn, XGBoost |
| Deployment    | Streamlit             |

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── eda_feature_engineering.py
│   ├── advanced_model_training.py
│
├── models/
│   └── house_price_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   └── graphs/
│
├── images/
│   ├── app_ui.png
│   ├── prediction.png
│   ├── charts.png
│
├── requirements.txt
└── README.md
```

---

## 📸 Screenshots

### 🔹 App Dashboard

![Dashboard](images/house_prediction_ui.png)

### 🔹 dataset_preview

![dataset_preview](images/dataset_preview.png)

### 🔹 summary_statics

![summary_statics](images/summary_statics.png)

### 🔹 price_distribution

![price_distribution](images/price_distribution.png)

### 🔹 correlation_heatmap

![correlation_heatmap](images/correlation_heatmap.png)

### 🔹 price_comparison

![price_comparison](images/price_comparison.png)
---

## 📈 Model Performance

| Model             | R² Score |
| ----------------- | -------- |
| Linear Regression | ~0.78    |
| Random Forest     | ~0.88    |
| XGBoost           | ~0.90    |

👉 Best Model: **XGBoost / Random Forest**

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/shwetasingh199/House-Price-Prediction
cd House-Price-Prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run App

```bash
streamlit run app/streamlit_app.py
```

---

## 💡 Key Insights

* Price increases significantly with area
* Location plays a major role
* Price per sqft is a strong indicator
* Tree-based models outperform linear models

---

## 🔮 Future Improvements

* 📍 Location-based filtering
* 🗺️ Map integration
* ☁️ Cloud deployment
* 📄 PDF report generation

---

## 👨‍💻 Author

Shweta Singh

---

## ⭐ Show Your Support

If you like this project:
👉 Star ⭐ the repo
👉 Share it with others
👉 Use it in your portfolio

---
