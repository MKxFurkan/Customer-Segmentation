# Customer Segmentation using K-Means Clustering

## Project Overview

This project uses unsupervised machine learning to segment customers based on their income, purchasing behavior and engagement patterns.

The objective is to identify distinct customer groups that can help marketing teams develop more targeted marketing strategies and offers.

## Dataset

- **Dataset:** Customer Personality Analysis
- **Original records:** 2,240
- **Original features:** 29
- **Records after cleaning:** 2,232

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Data Cleaning and Feature Engineering

- Handled missing values through median imputation in the preprocessing pipeline.
- Removed irrelevant features.
- Created `Total_Spending` by combining product spending columns.
- Created `Total_Purchases` by combining web, catalog and store purchases.
- Created `Children` by combining `Kidhome` and `Teenhome`.
- Handled income outliers while preserving missing values for imputation.

## Feature Selection

Selected six features for clustering:

- Income
- Total_Spending
- Total_Purchases
- Children
- NumWebVisitsMonth
- NumDealsPurchases

## Preprocessing

Used Scikit-learn's `ColumnTransformer` and `Pipeline`:

- Median imputation for missing Income values.
- Log transformation (`log1p`) for Total_Spending.
- StandardScaler for all selected features.

## Clustering Methodology

- Implemented K-Means clustering.
- Evaluated cluster counts from K=2 to K=6.
- Compared K-Means with Agglomerative Hierarchical Clustering using Ward linkage.
- Selected K=3 as the candidate segmentation because it provides three interpretable customer groups for marketing analysis.
- Combined preprocessing and K-Means into a single Scikit-learn pipeline.

## Model Evaluation

K-Means evaluation results:

| K | Silhouette Score | Davies-Bouldin Score |
|---|---:|---:|
| 2 | 0.394 | 1.040 |
| 3 | 0.371 | 1.095 |
| 4 | 0.347 | 1.129 |
| 5 | 0.306 | 1.204 |
| 6 | 0.298 | 1.243 |

K=2 achieved the best numerical scores. K=3 was selected as a business-oriented candidate because it provides three distinguishable customer segments.

K-Means also performed better than Ward hierarchical clustering across the tested cluster counts.

## Customer Segments

| Segment | Customers | Average Income | Average Spending | Average Purchases | Response Rate |
|---|---:|---:|---:|---:|---:|
| Moderate-value customers | 555 | 55,199 | 694 | 16.31 | 13.87% |
| High-value customers | 737 | 73,313 | 1,208 | 18.73 | 23.47% |
| Low-spending customers | 940 | 33,227 | 81 | 5.46 | 8.72% |

### Business Recommendations

**1. Moderate-value customers**
- Offer discounts and product bundles.
- Encourage repeat purchases through engagement campaigns.

**2. High-value customers**
- Provide premium offers.
- Introduce loyalty rewards and exclusive deals.

**3. Low-spending customers**
- Offer entry-level products.
- Use targeted engagement campaigns to encourage purchases.

## Limitations

- Customer segments have some overlap.
- Cluster stability was not evaluated.
- Marketing recommendations are suggestions and have not been experimentally validated.
- K=3 was selected for business interpretability rather than the highest numerical clustering scores.

## Project Status

Core ML Pipeline Completed