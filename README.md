# 🛒 Customer Segmentation — E-Commerce Intelligence Platform

> **Unsupervised machine learning pipeline to identify distinct customer groups from transactional e-commerce data, enabling data-driven marketing and retention strategies.**

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0?style=flat)

---

## 📌 Project Overview

Most e-commerce businesses treat all customers identically — same emails, same discounts, same messaging. This project challenges that approach by applying **K-Means clustering** on enriched RFM (Recency, Frequency, Monetary) and behavioural features to surface meaningful customer segments from raw transactional data.

The outcome is an actionable segmentation model that marketing, CRM, and product teams can use to personalise customer experiences and allocate retention budgets efficiently.

**Key results at a glance:**

| Metric | Value |
|---|---|
| Customers analysed | 1,000 |
| Transactions processed | 13,740 |
| Features engineered | 21 |
| Optimal clusters (K) | 10 |
| Overall Silhouette Score | 0.35+ |
| Estimated total LTV | ₹13.46M |
| Premium segment share | 35% of customers, ~60% of revenue |

---

## 🗂️ Repository Structure

```
customer_segmentation/
│
├── data/
│   ├── generate_dataset.py           # Synthetic dataset generator (Faker + NumPy)
│   ├── raw_ecommerce_data.csv        # Raw transactional data (13,740 rows)
│   ├── customer_features.csv         # Engineered feature matrix (1 row/customer)
│   └── customer_segments_final.csv   # Final table with cluster labels
│
├── src/
│   ├── 01_eda.py                     # Exploratory Data Analysis
│   ├── 02_feature_engineering.py     # RFM + behavioural + category features
│   ├── 03_preprocessing_and_optimal_k.py  # Scaling, Elbow, Silhouette
│   ├── 04_kmeans_clustering.py       # K-Means training & evaluation
│   ├── 05_cluster_profiling.py       # Business labelling & profiling plots
│   └── 06_visualisations.py          # Final dashboard
│
├── outputs/
│   ├── cluster_summary.csv           # Per-cluster statistics report
│   ├── kmeans_model.pkl              # Trained K-Means model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── pca.pkl                       # Fitted PCA (2D projection)
│   ├── best_k.txt                    # Optimal K value
│   └── plots/
│       ├── 01_eda_overview.png
│       ├── 02_optimal_k.png
│       ├── 03_silhouette_plot.png
│       ├── 04_cluster_radar.png
│       ├── 05_cluster_boxplots.png
│       └── 06_final_dashboard.png
│
├── Customer_Segmentation_Analysis.ipynb   # Full analysis notebook (with outputs)
├── run_pipeline.py                        # One-command pipeline runner
└── README.md
```

---

## 🔬 Methodology

### 1. Data Generation
A realistic synthetic dataset was generated using `Faker` and `NumPy` with **5 seeded ground-truth segments** to mimic a real Indian e-commerce platform. The dataset spans 2 years (Jan 2023 – Dec 2024) across 8 product categories.

### 2. Exploratory Data Analysis
- Univariate distributions for order amount, recency, visit frequency
- Missing value audit (age: 4.8% missing, gender: 25% missing)
- Monthly revenue trend, category mix, payment method breakdown

### 3. Feature Engineering
Raw transactional data (1 row per order) was aggregated into a **customer-level feature matrix** (1 row per customer):

| Feature Group | Features |
|---|---|
| **RFM** | `recency`, `frequency`, `monetary_total`, `monetary_avg`, `spend_std` |
| **Behavioural** | `website_visits`, `spend_per_visit`, `purchase_rate` |
| **Category Affinity** | `cat_pref_electronics`, `cat_pref_fashion`, `cat_pref_home_and_kitchen`, `cat_pref_sports`, `cat_pref_beauty`, `cat_pref_books`, `cat_pref_toys`, `cat_pref_grocery` |

> Category features are expressed as **spend proportions** (not raw amounts) to separate preference from purchasing power.

### 4. Preprocessing
- **StandardScaler** applied to all 16 clustering features — essential for distance-based algorithms
- **Elbow Method** (K=2 to 12) to identify WCSS inflection point
- **Silhouette Score** used as the final selection criterion → **K = 10**
- **PCA** (2 components) computed for visualisation only (not for clustering)

### 5. K-Means Clustering
```
KMeans(n_clusters=10, init='k-means++', n_init=20, max_iter=300, random_state=42)
```
- `k-means++` initialisation reduces sensitivity to poor starting centroids
- `n_init=20` ensures stable results across random seeds
- Model, scaler, and PCA objects saved as `.pkl` files for reuse

### 6. Cluster Profiling
Clusters were labelled with business-readable names based on rule-based thresholds relative to dataset medians:

| Rule | Segment Label |
|---|---|
| `monetary_total > 1.4× median` | 💎 Premium Spenders |
| `frequency > 1.3× median` AND `monetary_avg < median` | 🛒 Frequent Budget Shoppers |
| `website_visits > 1.3× median` AND `purchase_rate < median` | 👀 Window Shoppers |
| `recency > 1.3× median` | 😴 Dormant Customers |
| All else | ⭐ Regular Loyal Buyers |

---

## 📊 Key Findings

### Segment Overview

| Cluster | Segment | Customers | Avg Total Spend | Avg Orders | Avg Recency |
|---|---|---|---|---|---|
| 3 | 💎 Premium Spenders | 150 (15%) | ₹66,091 | 22.9 | 33 days |
| 1 | 💎 Premium Spenders | 200 (20%) | ₹10,444 | 29.8 | 25 days |
| 0 | ⭐ Regular Loyal Buyers | 66 (6.6%) | ₹3,207 | 7.9 | 166 days |
| 5–9 | ⭐ Regular Loyal Buyers | ~396 (39.6%) | ₹1,700–₹2,700 | 5.6–7.2 | 110–164 days |
| 2, 4 | ⭐ Regular Loyal Buyers | 188 (18.8%) | ₹1,885–₹1,895 | 6.3–6.4 | 141–154 days |

### Revenue Insight
Despite representing only **35% of the customer base**, Premium Spenders (Clusters 1 & 3) contribute an estimated **~60% of total revenue** — a textbook illustration of the Pareto principle in e-commerce.

---

## 💡 Business Recommendations

### 💎 Premium Spenders
- Enrol in a **VIP loyalty tier** with early access to sales, free express shipping, and dedicated support
- Trigger **personalised reactivation campaigns** if recency exceeds 60 days
- Offer **curated bundles** in their top categories (Books, Beauty)

### ⭐ Regular Loyal Buyers
- Implement a **points-based rewards programme** to drive frequency uplift
- Use **category-specific promotions** tied to their dominant spend category
- A/B test push notification timing for restock and flash sale alerts

---

## ⚙️ How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib faker
```

### Option A — Full Pipeline (recommended)
```bash
python run_pipeline.py
```
Runs all 7 steps in order and saves all outputs automatically.

### Option B — Step by Step
```bash
python data/generate_dataset.py
python src/01_eda.py
python src/02_feature_engineering.py
python src/03_preprocessing_and_optimal_k.py
python src/04_kmeans_clustering.py
python src/05_cluster_profiling.py
python src/06_visualisations.py
```

### Option C — Jupyter Notebook
Open `Customer_Segmentation_Analysis.ipynb` in JupyterLab or VS Code.  
All outputs are pre-embedded — no need to re-run cells to see results.

---

## 🔮 Future Improvements

- [ ] **DBSCAN** pass to isolate outlier customers before K-Means
- [ ] **Time-series features** — purchase velocity, seasonal patterns, trend direction
- [ ] **SHAP explainability** — show which features drove each customer's cluster assignment
- [ ] **Monthly re-clustering pipeline** with cluster stability tracking over time
- [ ] **Propensity models** on top of segments — predict upgrade likelihood (Regular → Premium)
- [ ] **A/B testing framework** to measure revenue lift from segment-specific campaigns

---

## 👤 Author

**Siddharth Khedekar**  
Data Science Internship Project  

---

## 📄 License

This project is for educational and internship demonstration purposes.  
The dataset is fully synthetic — no real customer data was used.
