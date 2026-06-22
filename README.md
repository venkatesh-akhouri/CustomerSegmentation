# Customer Segmentation — Unsupervised ML for Retail Analytics

A machine learning project that applies unsupervised learning techniques to segment customers of an online retail business into actionable categories, enabling targeted marketing and improved customer retention strategies.

## Overview

Customer segmentation is a fundamental business intelligence challenge — understanding *who* your customers are and *how* they behave allows businesses to target the right customers with the right actions. This project tackles that challenge using a real-world retail dataset, combining RFM-based scoring with clustering algorithms to produce meaningful, interpretable segments.

## Dataset

- **Source:** UCI Online Retail Dataset (ID: 352)
- **Size:** 541,909 records
- **Content:** Transaction data from a UK-based online retailer (2010–2011), including invoice numbers, stock codes, descriptions, quantities, prices, customer IDs, and countries

## Methodology

### 1. Data Cleaning & Preprocessing
- Handled missing CustomerIDs using invoice-level mapping
- Filled missing product descriptions using StockCode dictionary
- Removed cancelled transactions and returns
- Dropped country column (dataset is predominantly UK-based)

### 2. Feature Engineering
- Extracted temporal features: day, month, year, hour, minute from InvoiceDate
- Created weekend/weekday feature to capture purchase timing behaviour
- Added `Total Amount` (UnitPrice × Quantity) and `PurchaseType` (Purchase/Return)

### 3. RFM-Based Segmentation
Calculated three core metrics per customer:
- **Recency** — how recently did the customer purchase?
- **Frequency** — how often do they purchase?
- **Monetary** — how much do they spend?

Applied quantile-based scoring to classify customers into segments:
- **Elite** — high frequency, high monetary, recent purchasers
- **Power** — above average across all dimensions
- **Regular** — average behaviour
- **Ghost** — inactive or low-value customers

### 4. Clustering
- **KMeans** — used elbow method to determine optimal K=6 clusters; applied StandardScaler for feature normalisation
- **DBSCAN** — evaluated as an alternative to identify density-based clusters and handle outliers
- Compared both approaches to identify the most appropriate segmentation strategy

## Key Insights

- Frequency and Monetary value showed a moderate positive correlation (r=0.46) — customers who purchase often also tend to spend more
- Recency was largely independent of other RFM metrics — a customer can be a high spender but lapse in activity
- **Strategic implication:** Targeting high-frequency customers is the strongest lever for revenue retention; reactivation campaigns should focus on high-monetary, low-recency customers separately

## Tech Stack

- **Python** — Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn (KMeans, DBSCAN, StandardScaler)
- **Data Source** — `ucimlrepo` library

## Results

| Approach | Clusters | Key Output |
|---|---|---|
| RFM Scoring | 4 segments | Elite, Power, Regular, Ghost |
| KMeans | K=6 | Optimal clusters via elbow method |
| DBSCAN | Density-based | Alternative clustering with outlier detection |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
jupyter notebook Project_Customer_Segmentation.ipynb
```

## Project Structure

```
├── Project_Customer_Segmentation.ipynb   # Main notebook
└── README.md
```

## Author

Venkatesh Akhouri  
MSc Image Analysis and Machine Learning, Uppsala University  
[GitHub](https://github.com/venkatesh-akhouri) | [LinkedIn](https://linkedin.com/in/venkateshakhouri)
