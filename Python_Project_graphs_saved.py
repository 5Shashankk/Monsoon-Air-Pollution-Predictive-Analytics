# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Automatically save all graphs in the DATASETS/Graphs folder
GRAPH_FOLDER = r"C:\Users\nawad\OneDrive\Desktop\INT234\DATASETS\Graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)

def save_graph(filename):
    filepath = os.path.join(GRAPH_FOLDER, filename + ".png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print("Graph saved:", filepath)
    plt.show()
    plt.close()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# -----------------------------------------------------------

# Step 1: Load datasets
rain = pd.read_excel(
    r"C:\Users\nawad\OneDrive\Desktop\INT234\All_India_Area_Weighted_Monthly_Seasonal_And_Annual_Rainfall.xlsx"
)

air = pd.read_excel(
    r"C:\Users\nawad\OneDrive\Desktop\INT234\India_NAMP_2011-2015_PM10_SO2_NO2.xlsx"
)

# Step 2: Explore datasets
print("\nRAINFALL DATASET")
print(rain.head())
print(rain.info())
print(rain.describe())

print("\nAIR POLLUTION DATASET")
print(air.head())
print(air.info())
print(air.describe())

print("\nRainfall columns:")
print(rain.columns.tolist())

print("\nAir pollution columns:")
print(air.columns.tolist())

# Step 3: Clean column names
rain.columns = rain.columns.astype(str).str.strip()
air.columns = air.columns.astype(str).str.strip()

# Step 4: Check missing values
print("\nMissing values in rainfall:")
print(rain.isnull().sum())

print("\nMissing values in air pollution:")
print(air.isnull().sum())

# Step 5: Prepare rainfall data
month_map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

rain_month_columns = list(month_map.keys())

rain["YEAR"] = pd.to_numeric(rain["YEAR"], errors="coerce")

for col in rain_month_columns + ["ANN"]:
    rain[col] = pd.to_numeric(rain[col], errors="coerce")

rain = rain.dropna(subset=["YEAR"]).copy()
rain["YEAR"] = rain["YEAR"].astype(int)

rain_long = rain.melt(
    id_vars=["YEAR", "ANN"],
    value_vars=rain_month_columns,
    var_name="month_name",
    value_name="rainfall_mm"
)

rain_long["month"] = rain_long["month_name"].map(month_map)

rain_long.rename(
    columns={
        "YEAR": "year",
        "ANN": "annual_rainfall_mm"
    },
    inplace=True
)

rain_long.drop(columns=["month_name"], inplace=True)

# Step 6: Prepare air pollution data
required_air_columns = [
    "year", "month", "SO2", "NO2", "PM10", "State", "City"
]

missing_columns = [
    col for col in required_air_columns if col not in air.columns
]

if missing_columns:
    raise ValueError(
        "These columns are missing from the air dataset: "
        + str(missing_columns)
    )

for col in ["year", "month", "SO2", "NO2", "PM10"]:
    air[col] = pd.to_numeric(air[col], errors="coerce")

air = air.dropna(subset=["year", "month"]).copy()

air["year"] = air["year"].astype(int)
air["month"] = air["month"].astype(int)

air = air[
    air["year"].between(2011, 2014)
    & air["month"].between(1, 12)
].copy()

# Step 7: Aggregate air pollution by city, year, and month
aggregation = {
    "SO2": "mean",
    "NO2": "mean",
    "PM10": "mean"
}

if "Stn Code" in air.columns:
    monthly_air = (
        air.groupby(
            ["year", "month", "State", "City"],
            as_index=False,
            dropna=False
        )
        .agg(
            SO2_mean=("SO2", "mean"),
            NO2_mean=("NO2", "mean"),
            PM10_mean=("PM10", "mean"),
            measurement_count=("PM10", "count"),
            station_count=("Stn Code", "nunique")
        )
    )
else:
    monthly_air = (
        air.groupby(
            ["year", "month", "State", "City"],
            as_index=False,
            dropna=False
        )
        .agg(
            SO2_mean=("SO2", "mean"),
            NO2_mean=("NO2", "mean"),
            PM10_mean=("PM10", "mean"),
            measurement_count=("PM10", "count")
        )
    )

# Step 8: Merge rainfall and air pollution datasets
df = monthly_air.merge(
    rain_long,
    on=["year", "month"],
    how="left",
    validate="many_to_one"
)

df["year_month"] = (
    df["year"].astype(str)
    + "-"
    + df["month"].astype(str).str.zfill(2)
)

print("\nMERGED DATASET")
print(df.head())
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Step 9: Save merged dataset
df.to_csv(
    r"C:\Users\nawad\OneDrive\Desktop\INT234\DATASETS\india_rainfall_air_quality_merged_2011_2014.csv",
    index=False
)

# -------------------------------------------
# 1: EXPLORING AND HANDLING DATA


print("\nPRACTICAL 1: DATA EXPLORATION")

print("\nDataset dimensions:")
print(df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nUnique cities:")
print(df["City"].nunique())

print("\nUnique states:")
print(df["State"].nunique())

print("\nYears available:")
print(sorted(df["year"].dropna().unique()))

print("\nMonthly pollution summary:")
print(
    df.groupby("month")[
        ["PM10_mean", "SO2_mean", "NO2_mean"]
    ].mean()
)

# Visualize monthly rainfall and pollution
monthly_summary = (
    df.groupby("month")[
        ["rainfall_mm", "PM10_mean", "SO2_mean", "NO2_mean"]
    ].mean()
)

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_summary.index,
    monthly_summary["rainfall_mm"],
    marker="o"
)
plt.xlabel("Month")
plt.ylabel("Average All-India Rainfall (mm)")
plt.title("Average Monthly Rainfall")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
save_graph("average_monthly_rainfall")

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_summary.index,
    monthly_summary["PM10_mean"],
    marker="o",
    label="PM10"
)
plt.plot(
    monthly_summary.index,
    monthly_summary["SO2_mean"],
    marker="s",
    label="SO2"
)
plt.plot(
    monthly_summary.index,
    monthly_summary["NO2_mean"],
    marker="^",
    label="NO2"
)
plt.xlabel("Month")
plt.ylabel("Average Pollutant Concentration")
plt.title("Average Monthly Air Pollution")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.tight_layout()
save_graph("average_monthly_air_pollution")

# ---------------------------------------
# 2: DATA PREPROCESSING


print("\nPRACTICAL 2: DATA PREPROCESSING")

numeric_columns = [
    "rainfall_mm",
    "annual_rainfall_mm",
    "PM10_mean",
    "SO2_mean",
    "NO2_mean"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows missing the target or essential features
df = df.dropna(
    subset=["rainfall_mm", "PM10_mean", "SO2_mean", "NO2_mean"]
).copy()

# Remove impossible negative values for rainfall and pollutants
df = df[
    (df["rainfall_mm"] >= 0)
    & (df["PM10_mean"] >= 0)
    & (df["SO2_mean"] >= 0)
    & (df["NO2_mean"] >= 0)
].copy()

# Encode categorical features
df["state_code"] = LabelEncoder().fit_transform(
    df["State"].fillna("Unknown").astype(str)
)

df["city_code"] = LabelEncoder().fit_transform(
    df["City"].fillna("Unknown").astype(str)
)

# Create seasonal features
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["season"] = df["month"].apply(assign_season)

df["season_code"] = LabelEncoder().fit_transform(df["season"])

print("\nCleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df[numeric_columns].isnull().sum())

# --------------------------------------
# CORRELATION ANALYSIS


print("\nCORRELATION ANALYSIS")

correlation_columns = [
    "rainfall_mm",
    "PM10_mean",
    "SO2_mean",
    "NO2_mean"
]

correlation_matrix = df[correlation_columns].corr()

print(correlation_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45
)
plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)
plt.title("Rainfall and Air Pollution Correlation")
plt.tight_layout()
save_graph("rainfall_pollution_correlation")

# ----------------------------------------------
# 3A: SIMPLE LINEAR REGRESSION


print("\nSIMPLE LINEAR REGRESSION")

simple_df = df.dropna(
    subset=["rainfall_mm", "PM10_mean"]
).copy()

X = simple_df[["rainfall_mm"]]
y = simple_df["PM10_mean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

y_pred = simple_model.predict(X_test)

print("Intercept:", simple_model.intercept_)
print("Coefficient:", simple_model.coef_[0])
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.scatter(X_test["rainfall_mm"], y_test, alpha=0.5)
plt.plot(
    X_test["rainfall_mm"],
    y_pred,
    linewidth=2
)
plt.xlabel("All-India Monthly Rainfall (mm)")
plt.ylabel("City-Month Average PM10")
plt.title("Simple Linear Regression: Rainfall vs PM10")
plt.tight_layout()
save_graph("simple_linear_regression_rainfall_vs_pm10")

# ------------------------------------------------------
# 3B: MULTIPLE LINEAR REGRESSION


print("\nMULTIPLE LINEAR REGRESSION")

features = [
    "rainfall_mm",
    "month",
    "year",
    "state_code",
    "city_code",
    "season_code"
]

target = "PM10_mean"

model_df = df.dropna(subset=features + [target]).copy()

X = model_df[features]
y = model_df[target]

# Time-based split to reduce future-to-past leakage
years = sorted(model_df["year"].unique())

if len(years) >= 2:
    split_position = max(1, int(len(years) * 0.75))
    split_position = min(split_position, len(years) - 1)
    test_start_year = years[split_position]

    train_mask = model_df["year"] < test_start_year
    test_mask = model_df["year"] >= test_start_year

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nRegression Coefficients:")
print("Intercept:", model.intercept_)

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)

print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

comparison = pd.DataFrame({
    "Actual PM10": y_test,
    "Predicted PM10": y_pred
})

print("\nActual vs Predicted:")
print(comparison.head(10))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linewidth=2
)

plt.xlabel("Actual PM10")
plt.ylabel("Predicted PM10")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.tight_layout()
save_graph("multiple_regression_actual_vs_predicted_pm10")

# ------------------------------------------------
# 3C: POLYNOMIAL REGRESSION


print("\nPOLYNOMIAL REGRESSION")

poly_df = df.dropna(
    subset=["rainfall_mm", "PM10_mean"]
).copy()

X = poly_df[["rainfall_mm"]]
y = poly_df["PM10_mean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

rain_grid = pd.DataFrame({
    "rainfall_mm": np.linspace(
        X["rainfall_mm"].min(),
        X["rainfall_mm"].max(),
        200
    )
})

plt.figure(figsize=(8, 5))
plt.scatter(X_test["rainfall_mm"], y_test, alpha=0.4)
plt.plot(
    rain_grid["rainfall_mm"],
    poly_model.predict(rain_grid),
    linewidth=2
)
plt.xlabel("Rainfall (mm)")
plt.ylabel("PM10")
plt.title("Polynomial Regression: Rainfall vs PM10")
plt.tight_layout()
save_graph("polynomial_regression_rainfall_vs_pm10")

# ----------------------------------------------------
# 3D: LOGISTIC REGRESSION


print("\nLOGISTIC REGRESSION")

# Define a relative high-pollution label within each city.
# This is a project-defined label, not an official AQI category.
class_df = df.dropna(
    subset=["PM10_mean", "rainfall_mm"]
).copy()

city_thresholds = class_df.groupby("City")["PM10_mean"].transform(
    "median"
)

class_df["high_PM10"] = (
    class_df["PM10_mean"] > city_thresholds
).astype(int)

classification_features = [
    "rainfall_mm",
    "month",
    "year",
    "state_code",
    "city_code",
    "season_code"
]

X = class_df[classification_features]
y = class_df["high_PM10"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression

log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))



# ---------------------------------------------------
# 4A: K-NEAREST NEIGHBORS


print("\nK-NEAREST NEIGHBORS CLASSIFICATION")

X = class_df[classification_features]
y = class_df["high_PM10"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------
# 4B: GAUSSIAN NAIVE BAYES


print("\nGAUSSIAN NAIVE BAYES CLASSIFICATION")

nb_model = make_pipeline(
    StandardScaler(),
    GaussianNB()
)

nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
y_prob = nb_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# ---------------------------------------------
# 5A: DECISION TREE CLASSIFICATION


print("\nDECISION TREE CLASSIFICATION")

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
y_prob = tree_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



plt.figure(figsize=(16, 8))
plot_tree(
    tree_model,
    feature_names=classification_features,
    class_names=["Normal-relative PM10", "High-relative PM10"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree for Relative PM10 Classification")
plt.tight_layout()
save_graph("decision_tree_pm10_classification")

# -------------------------------------------------------
# 5B: SUPPORT VECTOR MACHINE


print("\nSUPPORT VECTOR MACHINE CLASSIFICATION")

svm_model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# --------------------------------------------
# 6A: K-MEANS CLUSTERING


print("\nK-MEANS CLUSTERING")

cluster_features = [
    "rainfall_mm",
    "PM10_mean",
    "SO2_mean",
    "NO2_mean"
]

cluster_df = df.dropna(subset=cluster_features).copy()

X_cluster = cluster_df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow method to choose number of clusters
inertias = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.tight_layout()
save_graph("kmeans_elbow_method")

# Fit final K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(cluster_df["cluster"].value_counts().sort_index())

print("\nCluster characteristics:")
print(
    cluster_df.groupby("cluster")[cluster_features].mean()
)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_df["cluster"],
    alpha=0.6
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters Visualized with PCA")
plt.tight_layout()
save_graph("kmeans_clusters_pca")

print(
    "PCA explained variance:",
    pca.explained_variance_ratio_
)

# -----------------------------------------------
# 6B: HIERARCHICAL CLUSTERING


print("\nHIERARCHICAL CLUSTERING")

# Use a sample to keep hierarchical clustering computationally manageable
sample_size = min(2000, len(cluster_df))

hier_sample = cluster_df.sample(
    n=sample_size,
    random_state=42
).copy()

X_hier = scaler.fit_transform(hier_sample[cluster_features])

hier_model = AgglomerativeClustering(n_clusters=3)

hier_sample["hierarchical_cluster"] = hier_model.fit_predict(
    X_hier
)

print("\nHierarchical cluster counts:")
print(
    hier_sample["hierarchical_cluster"]
    .value_counts()
    .sort_index()
)

print("\nHierarchical cluster characteristics:")
print(
    hier_sample.groupby("hierarchical_cluster")[cluster_features].mean()
)

plt.figure(figsize=(8, 6))
plt.scatter(
    hier_sample["rainfall_mm"],
    hier_sample["PM10_mean"],
    c=hier_sample["hierarchical_cluster"],
    alpha=0.6
)
plt.xlabel("All-India Rainfall (mm)")
plt.ylabel("City-Month PM10")
plt.title("Hierarchical Clustering")
plt.tight_layout()
save_graph("hierarchical_clustering")

# --------------------------------------------------
# FINAL SUMMARY


print("\nPROJECT COMPLETED")
print("Project: Monsoon and Air Pollution Predictive Analytics")
print("Analysis period: 2011-2014")
print("Regression target: PM10 concentration")
print("Classification target: Relative high-PM10 label")
print("Clustering: K-Means and Hierarchical Clustering")

