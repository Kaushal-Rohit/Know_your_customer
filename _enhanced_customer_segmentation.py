import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DATA LOADING & INITIAL PREPROCESSING
# ============================================================================

c = pd.read_csv("smartcart_customers.csv")

# Handle missing values
c["Income"] = c["Income"].fillna(c["Income"].median())

# Feature Engineering
c["Age"] = 2026 - c["Year_Birth"]
c["Dt_Customer"] = pd.to_datetime(c["Dt_Customer"], dayfirst=True)
reference = c["Dt_Customer"].max()
c["JoinDate"] = (reference - c["Dt_Customer"]).dt.days

# Spending aggregation
spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", 
                 "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
c["Total_Spending"] = c[spending_cols].sum(axis=1)
c["Avg_Spending_Per_Month"] = c["Total_Spending"] / (c["JoinDate"] + 1)

# Children aggregation
c["Total_Children"] = c["Kidhome"] + c["Teenhome"]

# Education standardization
c["Education"] = c["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate", "PhD": "Postgraduate"
})

# Living situation
c["Living_With"] = c["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Single": "Alone", "Divorced": "Alone",
    "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
})

# Additional features for better clustering
c["Response_Rate"] = (c["Response"] / (c["NumDealsPurchases"] + 1)) * 100
c["Web_Purchase_Ratio"] = c["NumWebPurchases"] / (c["NumWebPurchases"] + c["NumCatalogPurchases"] + c["NumStorePurchases"] + 1)

# ============================================================================
# 2. OUTLIER REMOVAL & CLEANING
# ============================================================================

print("=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)
print(f"Initial data size: {len(c)}")

# Remove outliers
c_cleaned = c[
    (c["Age"] < 90) & 
    (c["Age"] > 18) &
    (c["Income"] < 600_000) & 
    (c["Total_Children"] < 5)
].copy()

print(f"Data size after removing outliers: {len(c_cleaned)}")
print(f"Records removed: {len(c) - len(c_cleaned)}")
print("\n")

# ============================================================================
# 3. FEATURE SELECTION & ENCODING
# ============================================================================

# Select features for clustering
feature_cols = [
    "Income", "Recency", "Response", "Age", "Total_Spending", 
    "Total_Children", "Avg_Spending_Per_Month", "Response_Rate",
    "Web_Purchase_Ratio", "NumWebPurchases", "NumCatalogPurchases"
]

# Categorical encoding
cat_cols = ["Education", "Living_With"]
ohe = OneHotEncoder(sparse_output=False)
enc_cols = ohe.fit_transform(c_cleaned[cat_cols])
enc_df = pd.DataFrame(
    enc_cols, 
    columns=ohe.get_feature_names_out(cat_cols), 
    index=c_cleaned.index
)

# Combine numerical and categorical features
c_encoded = pd.concat([c_cleaned[feature_cols], enc_df], axis=1)

# ============================================================================
# 4. STANDARDIZATION & PCA
# ============================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(c_encoded)

# PCA for dimensionality reduction
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("=" * 70)
print("PCA ANALYSIS")
print("=" * 70)
print(f"Variance explained by first 3 components: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Component 1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Component 2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Component 3: {pca.explained_variance_ratio_[2]:.2%}")
print("\n")

# ============================================================================
# 5. OPTIMAL CLUSTER DETERMINATION (ELBOW METHOD & SILHOUETTE)
# ============================================================================

print("=" * 70)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Curve & Silhouette Scores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method For Optimal K', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette scores
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score For Optimal K', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# Determine optimal k
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters (by Silhouette): {optimal_k}")
print(f"Silhouette score: {max(silhouette_scores):.4f}")

# ============================================================================
# 6. FINAL CLUSTERING WITH OPTIMAL K
# ============================================================================

print("\n" + "=" * 70)
print(f"APPLYING K-MEANS CLUSTERING WITH K={optimal_k}")
print("=" * 70)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
c_cleaned['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nCluster distribution:")
print(c_cleaned['Cluster'].value_counts().sort_index())
print("\n")

# ============================================================================
# 7. VISUALIZATION - 3D PCA with Clusters
# ============================================================================

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for cluster in range(optimal_k):
    cluster_data = X_pca[c_cleaned['Cluster'] == cluster]
    ax.scatter(
        cluster_data[:, 0], 
        cluster_data[:, 1], 
        cluster_data[:, 2],
        c=[colors[cluster]], 
        label=f'Cluster {cluster}',
        s=100, 
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11)
ax.set_title('Customer Segments in PCA Space', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

plt.savefig('clusters_3d_pca.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. DETAILED BEHAVIORAL ANALYSIS FOR EACH CLUSTER
# ============================================================================

print("=" * 70)
print("DETAILED CLUSTER BEHAVIORAL ANALYSIS")
print("=" * 70 + "\n")

analysis_features = [
    'Income', 'Age', 'Total_Spending', 'Total_Children', 
    'Recency', 'Response', 'JoinDate', 'Avg_Spending_Per_Month',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
]

cluster_analysis = pd.DataFrame()

for cluster in sorted(c_cleaned['Cluster'].unique()):
    cluster_data = c_cleaned[c_cleaned['Cluster'] == cluster]
    
    print(f"\n{'=' * 70}")
    print(f"CLUSTER {cluster} PROFILE")
    print(f"{'=' * 70}")
    print(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(c_cleaned)*100:.1f}%)")
    print(f"\n{'DEMOGRAPHICS':<30}")
    print(f"{'─' * 70}")
    print(f"  Average Age: {cluster_data['Age'].mean():.1f} years (±{cluster_data['Age'].std():.1f})")
    print(f"  Average Income: ${cluster_data['Income'].mean():,.0f} (±${cluster_data['Income'].std():,.0f})")
    print(f"  Average Children: {cluster_data['Total_Children'].mean():.2f}")
    
    print(f"\n{'SPENDING BEHAVIOR':<30}")
    print(f"{'─' * 70}")
    print(f"  Total Avg Spending: ${cluster_data['Total_Spending'].mean():,.0f}")
    print(f"  Monthly Avg Spending: ${cluster_data['Avg_Spending_Per_Month'].mean():.2f}")
    print(f"  Avg Spending on Wines: ${cluster_data['MntWines'].mean():,.0f}")
    print(f"  Avg Spending on Meat: ${cluster_data['MntMeatProducts'].mean():,.0f}")
    print(f"  Avg Spending on Fish: ${cluster_data['MntFishProducts'].mean():,.0f}")
    
    print(f"\n{'ENGAGEMENT METRICS':<30}")
    print(f"{'─' * 70}")
    print(f"  Response Rate: {cluster_data['Response'].mean()*100:.1f}%")
    print(f"  Days Since Last Purchase: {cluster_data['Recency'].mean():.0f}")
    print(f"  Avg Days as Customer: {cluster_data['JoinDate'].mean():.0f}")
    print(f"  Web Purchases: {cluster_data['NumWebPurchases'].mean():.1f}")
    print(f"  Catalog Purchases: {cluster_data['NumCatalogPurchases'].mean():.1f}")
    print(f"  Store Purchases: {cluster_data['NumStorePurchases'].mean():.1f}")
    
    print(f"\n{'EDUCATION & FAMILY COMPOSITION':<30}")
    print(f"{'─' * 70}")
    education_dist = cluster_data['Education'].value_counts()
    for edu, count in education_dist.items():
        print(f"  {edu}: {count} ({count/len(cluster_data)*100:.1f}%)")
    
    living_dist = cluster_data['Living_With'].value_counts()
    print(f"\n  Family Status:")
    for living, count in living_dist.items():
        print(f"    {living}: {count} ({count/len(cluster_data)*100:.1f}%)")
    
    # Store for comparison dataframe
    cluster_analysis = pd.concat([
        cluster_analysis,
        pd.DataFrame({
            'Cluster': [cluster],
            'Size': [len(cluster_data)],
            'Avg_Income': [cluster_data['Income'].mean()],
            'Avg_Age': [cluster_data['Age'].mean()],
            'Total_Spending': [cluster_data['Total_Spending'].mean()],
            'Response_Rate': [cluster_data['Response'].mean()],
            'Recency': [cluster_data['Recency'].mean()],
            'Monthly_Spending': [cluster_data['Avg_Spending_Per_Month'].mean()]
        })
    ], ignore_index=True)

# ============================================================================
# 9. COMPARATIVE VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Income comparison
axes[0, 0].bar(cluster_analysis['Cluster'], cluster_analysis['Avg_Income'], color=colors)
axes[0, 0].set_title('Average Income by Cluster', fontweight='bold')
axes[0, 0].set_ylabel('Income ($)')
axes[0, 0].set_xlabel('Cluster')

# Age comparison
axes[0, 1].bar(cluster_analysis['Cluster'], cluster_analysis['Avg_Age'], color=colors)
axes[0, 1].set_title('Average Age by Cluster', fontweight='bold')
axes[0, 1].set_ylabel('Age (years)')
axes[0, 1].set_xlabel('Cluster')

# Total spending
axes[0, 2].bar(cluster_analysis['Cluster'], cluster_analysis['Total_Spending'], color=colors)
axes[0, 2].set_title('Average Total Spending by Cluster', fontweight='bold')
axes[0, 2].set_ylabel('Spending ($)')
axes[0, 2].set_xlabel('Cluster')

# Response rate
axes[1, 0].bar(cluster_analysis['Cluster'], cluster_analysis['Response_Rate']*100, color=colors)
axes[1, 0].set_title('Response Rate by Cluster', fontweight='bold')
axes[1, 0].set_ylabel('Response Rate (%)')
axes[1, 0].set_xlabel('Cluster')

# Monthly spending
axes[1, 1].bar(cluster_analysis['Cluster'], cluster_analysis['Monthly_Spending'], color=colors)
axes[1, 1].set_title('Average Monthly Spending by Cluster', fontweight='bold')
axes[1, 1].set_ylabel('Monthly Spending ($)')
axes[1, 1].set_xlabel('Cluster')

# Cluster sizes
axes[1, 2].bar(cluster_analysis['Cluster'], cluster_analysis['Size'], color=colors)
axes[1, 2].set_title('Cluster Size Distribution', fontweight='bold')
axes[1, 2].set_ylabel('Number of Customers')
axes[1, 2].set_xlabel('Cluster')

plt.tight_layout()
plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

for idx, row in cluster_analysis.iterrows():
    cluster = int(row['Cluster'])
    cluster_data = c_cleaned[c_cleaned['Cluster'] == cluster]
    
    print(f"\n{'CLUSTER ' + str(cluster) + ': ', 'ACTION ITEMS':<70}")
    print(f"{'─' * 70}")
    
    # Generate actionable insights
    if row['Avg_Income'] > cluster_analysis['Avg_Income'].quantile(0.75):
        print("✓ HIGH-VALUE SEGMENT: Focus on premium products and exclusive offerings")
    elif row['Avg_Income'] < cluster_analysis['Avg_Income'].quantile(0.25):
        print("✓ BUDGET SEGMENT: Offer value bundles and loyalty discounts")
    
    if row['Response_Rate'] > cluster_analysis['Response_Rate'].quantile(0.75):
        print("✓ HIGHLY ENGAGED: Leverage for referral programs and VIP benefits")
    elif row['Response_Rate'] < cluster_analysis['Response_Rate'].quantile(0.25):
        print("✓ LOW ENGAGEMENT: Re-engagement campaigns with special incentives needed")
    
    if row['Recency'] > cluster_analysis['Recency'].quantile(0.75):
        print("✓ AT-RISK SEGMENT: Immediate win-back campaigns recommended")
    else:
        print("✓ LOYAL SEGMENT: Maintain engagement through regular communication")
    
    if row['Total_Spending'] > cluster_analysis['Total_Spending'].quantile(0.75):
        print("✓ HIGH SPENDERS: Cross-sell and upsell opportunities")
    
    if cluster_data['Total_Children'].mean() > 1:
        print("✓ FAMILY-ORIENTED: Family packages and bulk discounts effective")

print("\n" + "=" * 70)