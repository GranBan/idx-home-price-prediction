# California Residential Home Price Prediction

Machine learning model to predict California single-family residential home close prices, built as part of the IDX Exchange Data Science Internship Program.

**Live demo:** [Streamlit App](https://idx-home-price-prediction-app.streamlit.app/)
**Deployment repo:** [idx-home-price-prediction-streamlit](https://github.com/GranBan/idx-home-price-prediction-streamlit)

---

## Overview

This project predicts the close price of California single-family homes using real CRMLS MLS data, sourced via IDX Exchange's FTP feed. The pipeline covers full exploratory data analysis, data cleaning, geospatial feature engineering, and a tuned XGBoost model, evaluated against Linear Regression and Random Forest baselines.

**Final model results** (held out test month, June 2026):

| Metric | Value |
|---|---|
| R² | 0.9115 |
| MAPE | 0.1138 |
| MdAPE | 0.0719 |

---

## Dataset

- **Source:** CRMLS (California Regional Multiple Listing Service) MLS data, accessed via IDX Exchange FTP feed
- **Scope:** Single Family Residential properties in California
- **Time range:** June 2023 – June 2026 (effective dataset; earlier months excluded due to incomplete data ingestion, confirmed during EDA)
- **Final size:** ~397,000 records after cleaning, 24 features

Raw data is not included in this repository per program data use requirements. See [Reproducing This Project](#reproducing-this-project) below for access instructions.

---

## Repository Structure

```
IDXExchange/
├── 01_exploration.ipynb EDA: distributions, correlations, temporal trends
├── 02_preprocessing.ipynb Cleaning, outlier removal, geocoding, feature engineering
├── 03_baseline_model.ipynb Linear Regression baseline
├── 04_model_comparison.ipynb Random Forest model
├── 05_feature_engineering_geo.ipynb School district spatial join
├── 06_geo_clustering.ipynb K-means geographic clustering
├── 07_xgboost_model.ipynb XGBoost model, versions A/B/C with documented iterations
├── 08_evaluation_explainability.ipynb Feature importance, SHAP, price band analysis, deployment checks
└── README.md
```


---

## Pipeline Summary

### 1. Exploratory Data Analysis
Identified that ClosePrice is heavily right-skewed, raw numeric features (bedrooms, bathrooms, living area) show weak correlation with price, and records prior to June 2023 have suspiciously low monthly transaction counts, traced to incomplete data ingestion rather than real market conditions.

### 2. Preprocessing
- Filtered to `PropertyType = Residential`, `PropertySubType = SingleFamilyResidence`, California only
- Dropped leakage columns (`ListPrice`, `OriginalListPrice`, `DaysOnMarket`, `PurchaseContractDate`)
- Removed outliers below $10,000 and above $5,000,000 (percentile-driven, not arbitrary)
- Geocoded 34 records with missing coordinates using address lookup rather than median imputation
- Caught and fixed a data quality bug: 91 records had invalid lat/long values outside California's real geographic bounds, silently corrupting downstream clustering until identified
- Engineered `PropertyAge`, `BedBathRatio`, `CloseMonth`, `CloseYear`, `ListingDuration`
- Label-encoded high-cardinality categoricals (County, City, PostalCode), with encoders saved for deployment reuse

### 3. Feature Engineering
- **School district spatial join:** true point-in-polygon join (GeoPandas) against California Unified School District boundaries, replacing `HighSchoolDistrict` (25% missing) with a complete `DistrictName` feature
- **Geographic clustering:** K-means on property coordinates, cluster count (k=10) selected using both the elbow method and direct price-separation testing across k=2–20

### 4. Modeling

| Model | R² | MAPE | MdAPE |
|---|---|---|---|
| Linear Regression (baseline) | 0.5468 | 0.4000 | 0.2699 |
| Random Forest (default) | 0.9033 | 0.1268 | 0.0736 |
| **XGBoost (final, Version C)** | **0.9115** | **0.1138** | **0.0719** |

XGBoost was iterated through three documented versions:
- **Version A:** default hyperparameters (baseline, underperforms Random Forest as expected)
- **Version B:** tuned depth, learning rate, subsampling (surpasses Random Forest)
- **Version C:** validated training window length (confirmed full history outperforms shorter recent windows), applied log-transformation to the target, tested and rejected monotonic constraints and additional regularization (documented tradeoffs), and used early stopping to automatically determine optimal tree count

### 5. Evaluation & Explainability
- **Feature importance:** `GeoCluster` is the dominant feature (importance 0.54), over 6x the next feature, directly confirming the EDA hypothesis that location drives price more than raw property specs
- **SHAP analysis:** Latitude/Longitude show the largest individual prediction impact, with lower values (Southern California, coastal areas) consistently pushing predicted price up
- **Price band analysis:** model performs best on $500K–$1M homes (MdAPE 5.7%), with accuracy degrading at both extremes, luxury homes above $2M show the highest error (MdAPE 10.6%), consistent with sparser, more idiosyncratic pricing at that tier
- **Deployment readiness:** ~21 MB model file, ~1ms average inference latency per prediction

---

## Deployment

A live interactive Streamlit app is deployed separately, allowing users to enter an address or click a location on a map to get a real-time price estimate. See the [deployment repository](https://github.com/GranBan/idx-home-price-prediction-streamlit) for that code.

---

## Reproducing This Project

1. Request access to the CRMLS data FTP feed through the IDX Exchange internship program
2. Download the monthly `CRMLSSold*.csv` files locally (not included in this repo)
3. Update the file path in `01_exploration.ipynb` and `02_preprocessing.ipynb` to point to your local data directory
4. Install dependencies:
```bash
   pip install pandas numpy scikit-learn xgboost geopandas shapely geopy shap matplotlib seaborn
```
5. Run notebooks in order: `01` → `02` → `05` → `06` → `03`/`04` → `07` → `08`

Each notebook is self-contained and reloads data from the previous stage's saved output, so they can be rerun independently once earlier stages have completed at least once.

---

## Key Learnings

- Location dominates price prediction in California real estate far more than raw property specs like square footage or bedroom count
- Data quality issues can silently corrupt downstream features; a coordinate validity check caught 91 records that would have degraded geographic clustering
- Structural feature importance and SHAP-based prediction impact can disagree, both are worth examining for a complete picture of model behavior
- Model accuracy is not uniform across price tiers; luxury properties are meaningfully harder to predict than mid-market homes

---

Built as part of the IDX Exchange Data Science Internship Program, Summer/Fall 2026.
