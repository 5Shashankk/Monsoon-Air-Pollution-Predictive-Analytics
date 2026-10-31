# 🌧️ Monsoon and Air Pollution Predictive Analytics

### Data Science Minor Project | Lovely Professional University (LPU)

A data science and machine learning project that explores the relationship between monsoon rainfall and air pollution levels in India. The project uses historical rainfall and air-quality datasets to perform exploratory data analysis, predictive modelling, and clustering.

---

## 📌 Project Overview

Air pollution is a major environmental concern in India. Rainfall may influence the concentration of air pollutants through atmospheric processes such as pollutant washout.

This project analyzes rainfall and air pollution data to understand their relationship and explore whether machine learning models can predict and classify pollution levels.

The project focuses on three major pollutants:

- PM10
- SO2
- NO2

The analysis uses merged monthly rainfall and city-level air-quality data for **2011–2014**.

## 🎯 Objectives

1. Analyze historical rainfall and air pollution patterns in India.
2. Study the relationship between rainfall and major air pollutants.
3. Perform data cleaning, preprocessing, and exploratory data analysis.
4. Apply regression algorithms to predict PM10 levels.
5. Apply classification algorithms to identify relatively high PM10 observations.
6. Use clustering techniques to identify pollution patterns.
7. Evaluate and compare machine learning models using performance metrics.

## 📂 Datasets

The project uses two Excel datasets:

| Dataset | Description |
|---|---|
| All India Area Weighted Monthly Seasonal and Annual Rainfall | Monthly and annual rainfall data for India |
| India NAMP 2011–2015 PM10 SO2 NO2 | Air-quality monitoring data containing PM10, SO2, NO2, city, state, year and month |

**Raw dataset information:**
- Rainfall dataset: 114 rows × 18 columns
- Air pollution dataset: 213,083 rows × 9 columns

**Analysis period:** 2011–2014

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Excel datasets
- Git and GitHub

## ⚙️ Project Workflow

1. Dataset collection and loading
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Rainfall and pollution data merging
5. Correlation analysis
6. Machine learning model implementation
7. Model evaluation
8. Data visualization
9. Clustering and pattern identification

## 🤖 Machine Learning Algorithms

### Regression Algorithms
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression

### Classification Algorithms
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

### Clustering Algorithms
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA) for cluster visualization

## 📊 Model Performance

### Regression Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Simple Linear Regression | 43.65 | 56.70 | 0.0120 |
| Multiple Linear Regression | 37.83 | 49.54 | 0.0993 |
| Polynomial Regression | 43.58 | 56.66 | 0.0134 |

**Best regression model:** Multiple Linear Regression, based on the lowest MAE and RMSE and highest R² among the tested regression models.

### Classification Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 63.52% | 60.18% | 78.04% | 0.6796 |
| KNN | 61.38% | 61.10% | 60.75% | 0.6093 |
| Gaussian Naive Bayes | 63.46% | 60.18% | 77.69% | 0.6782 |
| Decision Tree | 63.98% | 60.21% | 80.61% | 0.6893 |
| SVM | 63.40% | 60.39% | 76.05% | 0.6732 |

**Best classification model:** Decision Tree, with the highest accuracy, recall, and F1-score among the tested classifiers.

*Note: The high-PM10 class is defined using a relative, within-city median threshold. It is not an official AQI or regulatory pollution category.*

## 📈 Key Findings

- Rainfall has a weak negative correlation with PM10 (-0.1386).
- Rainfall has a weak negative correlation with SO2 (-0.0419).
- Rainfall has a weak negative correlation with NO2 (-0.0672).
- PM10 and NO2 show a positive correlation (0.4329).
- Multiple Linear Regression performs better than the other tested regression models.
- Decision Tree achieves the best classification performance among the tested classifiers.
- K-Means identifies three groups with different rainfall and pollution characteristics.

## 📁 Repository Structure

```text
Monsoon-Air-Pollution-Predictive-Analytics/
│
├── DATASETS/
│
├── All_India_Area_Weighted_Monthly_Seasonal_And_Annual_Rainfall.xlsx
├── India_NAMP_2011-2015_PM10_SO2_NO2.xlsx
├── Python_Project_graphs_saved.py
├── Project Report.docx
└── README.md
```

## ▶️ How to Run the Project

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Monsoon-Air-Pollution-Predictive-Analytics.git
```

**2. Open the project folder**

```bash
cd Monsoon-Air-Pollution-Predictive-Analytics
```

**3. Install the required libraries**

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

**4. Run the Python program**

```bash
python Python_Project_graphs_saved.py
```

Make sure the Excel dataset paths in the Python script match the location of the downloaded datasets.

## 🔮 Future Scope

- Extend the analysis using more recent rainfall and air-quality datasets.
- Include additional pollutants and meteorological variables.
- Explore advanced machine learning and deep learning techniques.
- Develop a dashboard for interactive pollution analysis.
- Improve prediction accuracy through feature engineering and model optimization.
- Perform region-specific and seasonal pollution analysis.

## ⚠️ Limitations

- The merged analysis covers 2011–2014.
- All-India rainfall is associated with city-level pollution observations, so it does not represent local city rainfall.
- The observed correlations do not establish causation.
- The regression models have relatively low R² scores, indicating limited predictive power.
- The classification target represents relative high PM10 observations, not official AQI categories.

## 👨‍💻 Project Information

**Project Title:** Monsoon and Air Pollution Predictive Analytics

**Project Type:** Data Science Minor Project

**Institution:** Lovely Professional University (LPU)

**Domain:** Data Science and Machine Learning

**Language:** Python

---

⭐ If you find this project useful, consider giving the repository a star!
