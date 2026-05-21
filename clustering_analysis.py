"""
Clustering Analysis Module
Performs K-Means and Hierarchical Clustering for buyer segmentation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def find_optimal_clusters(features, max_clusters=10):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score
    """
    print("\n--- Finding Optimal Number of Clusters ---")
    
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
        
        # Silhouette score
        sil_score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(sil_score)
        
        # Calinski-Harabasz score
        ch_score = calinski_harabasz_score(features, kmeans.labels_)
        calinski_scores.append(ch_score)
        
        print(f"K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil_score:.3f}, CH={ch_score:.0f}")
    
    # Find optimal K (max silhouette)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k} (based on silhouette score)")
    
    return optimal_k, inertia, silhouette_scores


def perform_kmeans_clustering(features, n_clusters):
    """
    Perform K-Means clustering
    """
    print(f"\n--- Performing K-Means Clustering with K={n_clusters} ---")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Calculate clustering metrics
    sil_score = silhouette_score(features, cluster_labels)
    db_score = davies_bouldin_score(features, cluster_labels)
    ch_score = calinski_harabasz_score(features, cluster_labels)
    
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Score: {db_score:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.2f}")
    
    return kmeans, cluster_labels


def perform_hierarchical_clustering(features, n_clusters):
    """
    Perform Hierarchical Clustering for validation
    """
    print(f"\n--- Performing Hierarchical Clustering with K={n_clusters} ---")
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hierarchical.fit_predict(features)
    
    sil_score = silhouette_score(features, cluster_labels)
    print(f"Hierarchical Clustering Silhouette Score: {sil_score:.4f}")
    
    return hierarchical, cluster_labels


def plot_dendrogram(features, save_path=None):
    """
    Plot dendrogram for hierarchical clustering visualization
    """
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(features[:100], method='ward')  # Sample for visualization
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    if save_path:
        plt.savefig(save_path)
        print(f"Dendrogram saved to {save_path}")
    plt.close()


def analyze_clusters(df, cluster_labels, cluster_names=None):
    """
    Analyze and interpret each cluster with detailed profiling
    """
    print("\n--- Cluster Analysis and Interpretation ---")
    
    df_analysis = df.copy()
    df_analysis['Cluster'] = cluster_labels
    
    cluster_profiles = {}
    
    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        n_clients = len(cluster_data)
        percentage = (n_clients / len(df_analysis)) * 100
        
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster}: {n_clients} clients ({percentage:.1f}%)")
        print(f"{'='*50}")
        
        # Demographic profile
        print("\n📊 DEMOGRAPHIC PROFILE:")
        if 'age' in cluster_data.columns:
            print(f"  • Average Age: {cluster_data['age'].mean():.1f} years")
        if 'gender' in cluster_data.columns:
            gender_dist = cluster_data['gender'].value_counts(normalize=True)
            print(f"  • Gender: {dict(gender_dist)}")
        if 'country' in cluster_data.columns:
            top_countries = cluster_data['country'].value_counts().head(3)
            print(f"  • Top Countries: {dict(top_countries)}")
        
        # Purchase behavior
        print("\n💰 PURCHASE BEHAVIOR:")
        if 'has_loan' in cluster_data.columns:
            loan_pct = cluster_data['has_loan'].mean() * 100
            print(f"  • Loan Utilization: {loan_pct:.1f}%")
        if 'is_investment' in cluster_data.columns:
            invest_pct = cluster_data['is_investment'].mean() * 100
            print(f"  • Investment Purpose: {invest_pct:.1f}%")
        if 'is_corporate' in cluster_data.columns:
            corporate_pct = cluster_data['is_corporate'].mean() * 100
            print(f"  • Corporate Buyers: {corporate_pct:.1f}%")
        if 'is_international' in cluster_data.columns:
            intl_pct = cluster_data['is_international'].mean() * 100
            print(f"  • International Buyers: {intl_pct:.1f}%")
        
        # Satisfaction
        print("\n⭐ CUSTOMER SATISFACTION:")
        if 'satisfaction_score' in cluster_data.columns:
            print(f"  • Average Score: {cluster_data['satisfaction_score'].mean():.2f}/5")
            sat_dist = cluster_data['satisfaction_score'].value_counts().sort_index()
            print(f"  • Score Distribution: {dict(sat_dist)}")
        
        # Referral channels
        print("\n📢 REFERRAL CHANNELS:")
        if 'referral_channel' in cluster_data.columns:
            channels = cluster_data['referral_channel'].value_counts(normalize=True).head(3)
            for channel, pct in channels.items():
                print(f"  • {channel}: {pct*100:.1f}%")
        
        # Store profile for later use
        profile = {
            'size': n_clients,
            'percentage': percentage,
            'avg_age': cluster_data['age'].mean() if 'age' in cluster_data.columns else None,
            'loan_rate': cluster_data['has_loan'].mean() if 'has_loan' in cluster_data.columns else None,
            'investment_rate': cluster_data['is_investment'].mean() if 'is_investment' in cluster_data.columns else None,
            'corporate_rate': cluster_data['is_corporate'].mean() if 'is_corporate' in cluster_data.columns else None,
            'international_rate': cluster_data['is_international'].mean() if 'is_international' in cluster_data.columns else None,
            'avg_satisfaction': cluster_data['satisfaction_score'].mean() if 'satisfaction_score' in cluster_data.columns else None,
            'top_channels': cluster_data['referral_channel'].value_counts().head(3).to_dict() if 'referral_channel' in cluster_data.columns else None
        }
        
        cluster_profiles[cluster] = profile
    
    return df_analysis, cluster_profiles


def assign_cluster_names(profiles):
    """
    Assign meaningful names to clusters based on their characteristics
    """
    cluster_names = {}
    
    for cluster, profile in profiles.items():
        # Determine buyer type based on profile characteristics
        is_corporate = profile['corporate_rate'] > 0.5 if profile['corporate_rate'] else False
        is_investment = profile['investment_rate'] > 0.7 if profile['investment_rate'] else False
        is_international = profile['international_rate'] > 0.5 if profile['international_rate'] else False
        uses_loan = profile['loan_rate'] > 0.6 if profile['loan_rate'] else False
        high_satisfaction = profile['avg_satisfaction'] > 4.5 if profile['avg_satisfaction'] else False
        avg_age = profile['avg_age'] if profile['avg_age'] else 40
        
        if is_corporate:
            cluster_names[cluster] = "🏢 Corporate Investors"
        elif is_international and is_investment:
            cluster_names[cluster] = "🌍 Global Investors"
        elif is_investment and high_satisfaction:
            cluster_names[cluster] = "💎 Luxury Investors"
        elif uses_loan and avg_age < 40:
            cluster_names[cluster] = "🏠 First-Time Buyers"
        elif is_investment:
            cluster_names[cluster] = "📈 Domestic Investors"
        elif avg_age > 50:
            cluster_names[cluster] = "👴 Retiree Buyers"
        else:
            cluster_names[cluster] = "👨‍👩‍👧‍👦 Standard Home Buyers"
    
    return cluster_names


def perform_pca_visualization(features, cluster_labels, save_path=None):
    """
    Perform PCA for 2D visualization of clusters
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('PCA Visualization of Buyer Segments')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    if save_path:
        plt.savefig(save_path)
        print(f"PCA plot saved to {save_path}")
    plt.close()
    
    return pca_result


def generate_segmentation_report(df_analysis, cluster_names):
    """
    Generate comprehensive segmentation report
    """
    print("\n" + "="*80)
    print("BUYER SEGMENTATION REPORT")
    print("="*80)
    
    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        name = cluster_names.get(cluster, f"Cluster {cluster}")
        
        print(f"\n{'─'*50}")
        print(f"{name}")
        print(f"{'─'*50}")
        print(f"Size: {len(cluster_data)} buyers ({len(cluster_data)/len(df_analysis)*100:.1f}%)")
        
        # Key characteristics
        print("\nKey Characteristics:")
        if 'age' in cluster_data.columns:
            print(f"  • Average Age: {cluster_data['age'].mean():.1f}")
        print(f"  • Investment Focus: {cluster_data['is_investment'].mean()*100:.1f}%")
        print(f"  • Loan Usage: {cluster_data['has_loan'].mean()*100:.1f}%")
        print(f"  • Corporate: {cluster_data['is_corporate'].mean()*100:.1f}%")
        print(f"  • International: {cluster_data['is_international'].mean()*100:.1f}%")
        
        # Top regions
        if 'region' in cluster_data.columns:
            print("\nTop Regions:")
            top_regions = cluster_data['region'].value_counts().head(3)
            for region, count in top_regions.items():
                print(f"  • {region}: {count} buyers")
        
        # Marketing recommendations
        print("\nMarketing Recommendations:")
        if name == "🌍 Global Investors":
            print("  • Target with international investment opportunities")
            print("  • Highlight ROI and cross-border benefits")
            print("  • Use multilingual marketing materials")
        elif name == "🏢 Corporate Investors":
            print("  • Focus on bulk purchase incentives")
            print("  • Offer corporate partnership programs")
            print("  • Highlight portfolio diversification benefits")
        elif name == "💎 Luxury Investors":
            print("  • Promote premium properties and exclusive listings")
            print("  • Offer VIP customer service")
            print("  • Focus on luxury amenities and locations")
        elif name == "🏠 First-Time Buyers":
            print("  • Emphasize financing options and loan assistance")
            print("  • Provide educational content about home buying")
            print("  • Offer starter home packages")
        elif name == "📈 Domestic Investors":
            print("  • Focus on local market growth potential")
            print("  • Highlight rental income opportunities")
            print("  • Offer property management services")
        else:
            print("  • Balance marketing across multiple channels")
            print("  • Focus on value proposition and location benefits")
            print("  • Offer competitive pricing and incentives")
    
    return


if __name__ == "__main__":
    print("Clustering Analysis Module Ready")
    print("Run app.py to perform full analysis with Streamlit dashboard")
