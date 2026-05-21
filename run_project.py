"""
Runner script to execute the entire project
Run this to generate all outputs and start the dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_data_preprocessing():
    """Run data preprocessing"""
    print("\nRunning data preprocessing...")
    from data_preprocessing import load_and_clean_data, clean_client_data, prepare_features_for_clustering, prepare_buyer_features
    
    clients, properties, merged = load_and_clean_data()
    clients_cleaned = clean_client_data(clients)
    features, _, _, _ = prepare_features_for_clustering(clients_cleaned)
    buyer_features = prepare_buyer_features(merged)
    
    print(f"Data preprocessing complete!")
    print(f"  - {len(clients_cleaned)} clients processed")
    print(f"  - {features.shape[1]} features created")
    print(f"  - {len(buyer_features)} unique buyers identified")
    
    return clients_cleaned, features, buyer_features

def run_clustering_analysis(clients_cleaned, features):
    """Run clustering analysis"""
    print("\nRunning clustering analysis...")
    from clustering_analysis import find_optimal_clusters, perform_kmeans_clustering, analyze_clusters, assign_cluster_names
    
    optimal_k, inertia, sil_scores = find_optimal_clusters(features)
    kmeans, cluster_labels = perform_kmeans_clustering(features, optimal_k)
    df_analysis, profiles = analyze_clusters(clients_cleaned, cluster_labels)
    cluster_names = assign_cluster_names(profiles)
    
    print(f"Clustering complete! Identified {optimal_k} segments:")
    for cluster, name in cluster_names.items():
        print(f"  - {name}")
    
    return df_analysis, profiles, cluster_names

def start_dashboard():
    """Start Streamlit dashboard"""
    print("\nStarting Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("="*60)
    print("Parcl Co. Limited - Buyer Segmentation Project")
    print("="*60)
    
    # Check if requirements are installed
    try:
        import streamlit
        print("Requirements already installed")
    except ImportError:
        install_requirements()
    
    # Run preprocessing
    clients_cleaned, features, buyer_features = run_data_preprocessing()
    
    # Run clustering
    df_analysis, profiles, cluster_names = run_clustering_analysis(clients_cleaned, features)
    
    # Save results
    df_analysis.to_csv('segmentation_results.csv', index=False)
    print("\nResults saved to 'segmentation_results.csv'")
    
    # Start dashboard
    start_dashboard()
