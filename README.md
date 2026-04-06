# 🛒 Know Your Customer — Smart Clustering using Unsupervised Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rohit%20Kaushal-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/kaushal-rohit-83911a1b6)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)]()

> **Author:** Rohit Kaushal Bharatbhai | KPGU Vadodara  
> **Contact:** stud.2201201708@kpgu.ac.in

---

## 📌 Project Overview

**Know Your Customer** is an unsupervised machine learning project that segments retail customers into distinct behavioral groups using **K-Means Clustering**. By analyzing demographics, spending habits, and engagement patterns from the SmartCart dataset, this project enables data-driven marketing decisions — helping businesses tailor strategies to the right customers.

The pipeline covers everything from raw data cleaning and feature engineering, all the way to cluster interpretation and business recommendations.

---

## 🗂️ Repository Structure

```
Know_your_customer/
│
├── smartcart_clustering.py       # Main Python script
├── smartcart_clustering.ipynb    # Jupyter Notebook (interactive version)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to ignore in Git
├── README.md                     # Project documentation
│
└── outputs/                      # Generated plots (after running)
    ├── optimal_clusters.png
    ├── clusters_3d_pca.png
    └── cluster_comparison.png
```

---

## ⚙️ How It Works — Pipeline

```
Raw CSV Data
     │
     ▼
Data Cleaning & Outlier Removal (2240 → 2236 records)
     │
     ▼
Feature Engineering
(Age, Total_Spending, Avg_Spending_Per_Month, Web_Purchase_Ratio, etc.)
     │
     ▼
One-Hot Encoding (Education, Living_With)
     │
     ▼
StandardScaler Normalization
     │
     ▼
PCA — 3 Components (46.90% variance explained)
     │
     ▼
K-Means Clustering (Optimal K=10, Silhouette=0.2040)
     │
     ▼
Cluster Analysis + Business Recommendations
```

---

## 📊 Dataset

The dataset used is `smartcart_customers.csv`, a customer personality/behaviour dataset containing:

| Feature | Description |
|---|---|
| `Year_Birth` | Customer's birth year |
| `Income` | Annual household income |
| `Education` | Education level |
| `Marital_Status` | Marital status |
| `Kidhome` / `Teenhome` | Number of children/teens at home |
| `Recency` | Days since last purchase |
| `MntWines`, `MntMeatProducts`, etc. | Spending on product categories |
| `NumWebPurchases`, `NumStorePurchases`, etc. | Purchase channel behaviour |
| `Response` | Response to last marketing campaign |

> **Note:** The CSV file is not included in this repo. You can use the [UCI Marketing Campaign dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) which this project is based on.

---

## 🔬 PCA Analysis

| Component | Variance Explained |
|---|---|
| PC1 | 22.07% |
| PC2 | 13.26% |
| PC3 | 11.57% |
| **Total (3 PCs)** | **46.90%** |

PCA is used purely for **3D visualization** of clusters. The actual K-Means clustering is performed on the full scaled feature space.

---

## 🔢 Optimal Cluster Selection

Both the **Elbow Method** and **Silhouette Score** were used to determine the optimal number of clusters.

| Metric | Value |
|---|---|
| Optimal K | **10** |
| Silhouette Score | **0.2040** |

---

## 👥 The 10 Customer Segments — Detailed Profiles

> Total customers analyzed: **2,236**

---

### 🔵 Cluster 0 — Senior Solo Postgrads, Moderate Wine Buyers
**Size:** 180 customers (8.1%)

| Attribute | Value |
|---|---|
| Avg Age | 60.0 years |
| Avg Income | $46,590 |
| Avg Children | 1.27 |
| Total Spending | $360 |
| Wine Spending | $231 |
| Response Rate | 0.0% |
| Days Since Purchase | 51 |
| Living With | Alone (100%) |
| Education | Postgraduate (100%) |

**Behaviour:** Older, highly educated singles with moderate income. Wine is their dominant spend (64% of total). Despite being active (recent purchases), they show zero campaign response. Store purchases are their preferred channel.

**💡 Strategy:** Curated wine subscription offers, loyalty rewards for in-store visits.

---

### 🟢 Cluster 1 — Mid-Income Graduate Couples, Low Spenders
**Size:** 368 customers (16.5%) — *Largest group*

| Attribute | Value |
|---|---|
| Avg Age | 55.3 years |
| Avg Income | $37,425 |
| Avg Children | 1.28 |
| Total Spending | $157 |
| Wine Spending | $74 |
| Response Rate | 0.0% |
| Days Since Purchase | 52 |
| Living With | Partner (100%) |
| Education | Graduate (100%) |

**Behaviour:** The largest segment. Graduate-educated couples with children and the lowest spending levels. All-channel usage is low; catalog purchases especially low. Zero campaign response suggests marketing fatigue or irrelevant targeting.

**💡 Strategy:** Family bundle deals, value-oriented promotions, re-engagement campaigns with meaningful incentives.

---

### 🟡 Cluster 2 — Affluent Senior Postgrads, Premium Shoppers
**Size:** 198 customers (8.9%)

| Attribute | Value |
|---|---|
| Avg Age | 61.5 years |
| Avg Income | $74,825 |
| Avg Children | 0.41 |
| Total Spending | $1,298 |
| Wine Spending | $713 |
| Meat Spending | $378 |
| Response Rate | 0.0% |
| Days Since Purchase | 53 |
| Living With | Partner (81.8%) |
| Education | Postgraduate (98.5%) |

**Behaviour:** High earners who spend heavily, especially on wine and meat. Nearly all are postgraduates. Despite high spending, campaign response is zero — they buy on their own terms. Catalog and store purchases are both high.

**💡 Strategy:** Premium product lines, exclusive member events, early access to new products. Do NOT push discount campaigns — they respond to exclusivity, not deals.

---

### 🔴 Cluster 3 — Top Earners, 100% Campaign Responders ⭐
**Size:** 171 customers (7.6%)

| Attribute | Value |
|---|---|
| Avg Age | 57.3 years |
| Avg Income | $78,913 |
| Avg Children | 0.13 |
| Total Spending | $1,557 |
| Wine Spending | $764 |
| Meat Spending | $499 |
| Response Rate | **100.0%** |
| Days Since Purchase | 40 |
| Avg Days as Customer | 422 |
| Education | Graduate (49.7%) / Postgraduate (43.3%) |

**Behaviour:** The most valuable cluster. Highest income, highest spending, most recently active, and 100% campaign response rate. Long-tenure customers (422 days avg). Very few children — disposable income is high. Multichannel shoppers.

**💡 Strategy:** VIP program, referral bonuses, upsell premium SKUs, first-look at new product lines. This segment is your brand's best advocate — treat them like it.

---

### 🟠 Cluster 4 — High-Income Graduate Couples, Heavy Web & Store Shoppers
**Size:** 298 customers (13.3%)

| Attribute | Value |
|---|---|
| Avg Age | 58.5 years |
| Avg Income | $69,153 |
| Avg Children | 0.64 |
| Total Spending | $1,158 |
| Wine Spending | $534 |
| Meat Spending | $338 |
| Response Rate | 0.3% |
| Days Since Purchase | 52 |
| Living With | Partner (90.9%) |
| Education | Graduate (100%) |

**Behaviour:** High-income graduate couples who spend significantly across categories. They are the top web purchasers (6.4 avg) and strong store shoppers. Nearly all live with a partner. Campaign response is near-zero despite high spending ability.

**💡 Strategy:** Omnichannel loyalty program, personalized online recommendations, targeted email campaigns rather than broad campaigns.

---

### ⚪ Cluster 5 — Brand New High-Value Customers (Micro Segment)
**Size:** 11 customers (0.5%) — *Smallest group*

| Attribute | Value |
|---|---|
| Avg Age | 58.5 years |
| Avg Income | $75,162 |
| Avg Children | 0.27 |
| Total Spending | $1,179 |
| Monthly Spending | **$354.93** |
| Response Rate | 9.1% |
| Avg Days as Customer | **3 days** |

**Behaviour:** An anomaly cluster of brand-new customers (avg 3 days tenure) who are already spending at a high rate — $354.93/month vs the average of $1–7. Their profile matches high-value segments but they've barely just joined.

**💡 Strategy:** Onboarding experience is critical here. First impressions will determine if they become Cluster 3 customers or churn. Assign dedicated onboarding flows, personalized welcome offers.

---

### 🟣 Cluster 6 — Undergrad Mixed-Income, Moderate Spenders
**Size:** 235 customers (10.5%)

| Attribute | Value |
|---|---|
| Avg Age | 51.8 years |
| Avg Income | $39,955 |
| Avg Children | 0.90 |
| Total Spending | $359 |
| Wine Spending | $135 |
| Meat Spending | $95 |
| Response Rate | 2.6% |
| Education | Undergraduate (100%) |

**Behaviour:** Younger than most clusters, undergrad-educated, moderate-income. Spend is spread across categories more evenly. Income variance is high ($21K std dev) suggesting a diverse group. Small but measurable campaign response (2.6%).

**💡 Strategy:** Mid-range product bundles, digital-first campaigns (slightly more responsive than similar clusters), price-sensitive promotions.

---

### 🔵 Cluster 7 — Graduate Singles, Mid-Spenders, Zero Engagement
**Size:** 301 customers (13.5%)

| Attribute | Value |
|---|---|
| Avg Age | 55.9 years |
| Avg Income | $48,117 |
| Avg Children | 1.01 |
| Total Spending | $440 |
| Wine Spending | $199 |
| Meat Spending | $120 |
| Response Rate | 0.0% |
| Living With | Alone (100%) |
| Education | Graduate (100%) |

**Behaviour:** Graduate-educated singles with moderate income and moderate spending. They buy consistently but ignore all campaigns. Similar profile to Cluster 0 but with more diverse spending patterns.

**💡 Strategy:** Behavioural triggers rather than campaign blasts — e.g., "You usually buy X around this time" nudges. Personalization over promotion.

---

### 🟤 Cluster 8 — Postgrad Couples, Lower Spenders, Wine Focused
**Size:** 319 customers (14.3%)

| Attribute | Value |
|---|---|
| Avg Age | 58.8 years |
| Avg Income | $44,093 |
| Avg Children | 1.34 |
| Total Spending | $237 |
| Wine Spending | $151 |
| Response Rate | 0.0% |
| Living With | Partner (100%) |
| Education | Postgraduate (100%) |

**Behaviour:** The second-largest cluster. Highly educated couples with children but low spending relative to income. Wine dominates their basket (64% of total spend). Very low catalog engagement (1.1 avg). Zero campaign response.

**💡 Strategy:** Wine loyalty rewards, family-oriented product suggestions, try push notifications or SMS over email campaigns.

---

### 🔶 Cluster 9 — Budget Loyalists, 100% Campaign Response ⭐
**Size:** 155 customers (6.9%)

| Attribute | Value |
|---|---|
| Avg Age | 56.4 years |
| Avg Income | $40,807 |
| Avg Children | 1.21 |
| Total Spending | $378 |
| Wine Spending | $225 |
| Response Rate | **100.0%** |
| Days Since Purchase | **30** |
| Avg Days as Customer | **473 days** |

**Behaviour:** The most loyal customers by tenure (473 days avg) and most recently active (30 days since last purchase). Despite modest income and spending, they respond to every campaign. Mixed education and living situation. These are the brand's loyal everyday shoppers.

**💡 Strategy:** Reward their loyalty with points programs, early sale access, and budget-friendly deals. They respond to campaigns — use that. A perfect segment for referral and word-of-mouth programs.

---

## 📈 Cluster Comparison at a Glance

| Cluster | Size | Avg Income | Total Spending | Response Rate | Key Trait |
|---|---|---|---|---|---|
| 0 | 180 (8.1%) | $46,590 | $360 | 0% | Solo postgrads, wine lovers |
| 1 | 368 (16.5%) | $37,425 | $157 | 0% | Largest, lowest spenders |
| 2 | 198 (8.9%) | $74,825 | $1,298 | 0% | Affluent premium buyers |
| **3** | 171 (7.6%) | **$78,913** | **$1,557** | **100%** | ⭐ Top VIP segment |
| 4 | 298 (13.3%) | $69,153 | $1,158 | 0.3% | High-income web shoppers |
| 5 | 11 (0.5%) | $75,162 | $1,179 | 9.1% | New high-value customers |
| 6 | 235 (10.5%) | $39,955 | $359 | 2.6% | Undergrad moderate spenders |
| 7 | 301 (13.5%) | $48,117 | $440 | 0% | Solo graduates, disengaged |
| 8 | 319 (14.3%) | $44,093 | $237 | 0% | Postgrad couples, wine focus |
| **9** | 155 (6.9%) | $40,807 | $378 | **100%** | ⭐ Loyal budget champions |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Kaushal-Rohit/Know_your_customer.git
cd Know_your_customer

# Install dependencies
pip install -r requirements.txt
```

### Running the Script

```bash
# Place your smartcart_customers.csv in the project root, then:
python smartcart_clustering.py
```

### Running the Notebook

```bash
jupyter notebook smartcart_clustering.ipynb
```

---

## 📦 Requirements

See `requirements.txt` for full list. Core libraries:

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — KMeans, PCA, StandardScaler, silhouette scoring
- `matplotlib`, `seaborn` — visualizations

---

## 🤝 Connect

Made with ❤️ by **Rohit Kaushal **  
 
🔗 [LinkedIn](https://www.linkedin.com/in/kaushal-rohit-83911a1b6)  
