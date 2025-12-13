# India Air Quality Analysis & Forecasting System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
Air pollution poses a critical public health challenge in India. This project analyzes historical air quality data (2015–2020) from 26 major cities to identify pollution patterns and build a machine learning solution for forecasting **PM2.5** levels.

The system processes raw data from multiple sources, performs extensive exploratory data analysis, and deploys a predictive model via an interactive web dashboard.

## 🚀 Key Features
* **Automated Data Pipeline:** Scripts merge and preprocess disparate datasets from 26 cities.
* **Exploratory Analysis (EDA):** In-depth visualization of seasonality, correlations, and outlier detection.
* **Machine Learning Pipeline:** Comparative analysis of 4 regression models, optimized via Hyperparameter Tuning.
* **Robust Preprocessing:** Uses **RobustScaler** to handle extreme pollution outliers without data loss.
* **Interactive Dashboard:** A **Streamlit** application for real-time pollution prediction based on user inputs.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-learn (Random Forest, Gradient Boosting)
* **Deployment:** Streamlit
* **Version Control:** Git & GitHub

## 📂 Repository Structure
```text
├── .gitignore             # Configuration for ignored files
├── application.py         # Main Streamlit Dashboard Application
├── eda.ipynb              # Jupyter Notebook for Exploratory Data Analysis
├── model_building.ipynb   # Jupyter Notebook for Model Training & Evaluation
├── readme.md              # Project Documentation
└── requirements.txt       # Project Dependencies


Bash
git clone https://github.com/yoursmaddyy/indian-air-pollution.git
cd india-air-quality-project
Install DependenciesBashpip install -r requirements.txt
Run the App
Bash
streamlit run application.py
📈 Future Scope
Integration of real-time weather API data.
Implementation of LSTM (Deep Learning) for time-series specific forecasting.
Cloud deployment.