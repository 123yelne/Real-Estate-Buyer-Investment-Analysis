"""
Streamlit Dashboard for Real Estate Buyer Segmentation
Interactive analytics for Parcl Co. Limited
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Parcl - Buyer Segmentation Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import preprocessing and clustering modules
from data_preprocessing import load_and_clean_data, clean_client_data, prepare_features_for_clustering, prepare_buyer_features
from clustering_analysis import find_optimal_clusters, perform_kmeans_clustering, analyze_clusters, assign_cluster_names, perform_pca_visualization, generate_segmentation_report

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5A88;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1E3A5F;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'cluster_profiles' not in st.session_state:
    st.session_state.cluster_profiles = None
if 'cluster_names' not in st.session_state:
    st.session_state.cluster_names = None
if 'df_analysis' not in st.session_state:
    st.session_state.df_analysis = None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x80?text=Parcl+Co.", use_column_width=True)
    st.markdown("## 🎯 Filters")
    
    # Load data
    @st.cache_data
    def load_data():
        clients, properties, merged = load_and_clean_data()
        clients_cleaned = clean_client_data(clients)
        return clients, properties, merged, clients_cleaned
    
    with st.spinner("Loading data..."):
        clients, properties, merged, clients_cleaned = load_data()
    
    st.success(f"✅ Loaded {len(clients_cleaned)} clients")
    
    # Sidebar filters for dashboard
    st.markdown("### 📊 Dashboard Filters")
    
    selected_countries = st.multiselect(
        "Country",
        options=sorted(clients_cleaned['country'].dropna().unique()),
        default=[]
    )
    
    selected_purposes = st.multiselect(
        "Acquisition Purpose",
        options=sorted(clients_cleaned['acquisition_purpose'].dropna().unique()),
        default=[]
    )
    
    selected_client_types = st.multiselect(
        "Client Type",
        options=sorted(clients_cleaned['client_type'].dropna().unique()),
        default=[]
    )
    
    min_age, max_age = st.slider(
        "Age Range",
        min_value=int(clients_cleaned['age'].min()) if not clients_cleaned['age'].isna().all() else 18,
        max_value=int(clients_cleaned['age'].max()) if not clients_cleaned['age'].isna().all() else 100,
        value=(25, 65)
    )
    
    st.markdown("---")
    st.markdown("### 🔬 Clustering Settings")
    
    n_clusters = st.selectbox("Number of Clusters", [3, 4, 5, 6, 7, 8], index=1)
    
    run_clustering = st.button("🚀 Run Segmentation Analysis", type="primary", use_container_width=True)

# Main content
st.markdown('<h1 class="main-header">🏠 Parcl Real Estate</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 2rem;">AI-Powered Buyer Segmentation & Investment Intelligence</p>', unsafe_allow_html=True)

# Run clustering when button is clicked
if run_clustering or st.session_state.clustering_done:
    
    if run_clustering:
        with st.spinner("Running clustering analysis..."):
            # Prepare features
            scaled_features, feature_matrix, scaler, feature_cols = prepare_features_for_clustering(clients_cleaned)
            
            # Perform clustering
            kmeans, cluster_labels = perform_kmeans_clustering(scaled_features, n_clusters)
            
            # Analyze clusters
            df_analysis, cluster_profiles = analyze_clusters(clients_cleaned, cluster_labels)
            
            # Assign cluster names
            cluster_names = assign_cluster_names(cluster_profiles)
            
            # Map names back to dataframe
            df_analysis['Cluster_Name'] = df_analysis['Cluster'].map(cluster_names)
            
            # Store in session state
            st.session_state.clustering_done = True
            st.session_state.cluster_labels = cluster_labels
            st.session_state.cluster_profiles = cluster_profiles
            st.session_state.cluster_names = cluster_names
            st.session_state.df_analysis = df_analysis
            st.session_state.scaled_features = scaled_features
            
            st.success(f"✅ Segmentation complete! Identified {n_clusters} distinct buyer segments.")
    
    else:
        df_analysis = st.session_state.df_analysis
        cluster_names = st.session_state.cluster_names
        cluster_labels = st.session_state.cluster_labels
        cluster_profiles = st.session_state.cluster_profiles
        scaled_features = st.session_state.scaled_features
    
    # Apply filters to data
    filtered_df = df_analysis.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_purposes:
        filtered_df = filtered_df[filtered_df['acquisition_purpose'].isin(selected_purposes)]
    if selected_client_types:
        filtered_df = filtered_df[filtered_df['client_type'].isin(selected_client_types)]
    filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]
    
    # ==================== METRICS ROW ====================
    st.markdown('<h2 class="sub-header">📈 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Buyers", f"{len(filtered_df):,}")
    with col2:
        investment_pct = filtered_df['is_investment'].mean() * 100 if 'is_investment' in filtered_df.columns else 0
        st.metric("Investment Purpose", f"{investment_pct:.1f}%")
    with col3:
        loan_pct = filtered_df['has_loan'].mean() * 100 if 'has_loan' in filtered_df.columns else 0
        st.metric("Loan Users", f"{loan_pct:.1f}%")
    with col4:
        satisfaction = filtered_df['satisfaction_score'].mean() if 'satisfaction_score' in filtered_df.columns else 0
        st.metric("Avg Satisfaction", f"{satisfaction:.1f}/5")
    with col5:
        intl_pct = filtered_df['is_international'].mean() * 100 if 'is_international' in filtered_df.columns else 0
        st.metric("International", f"{intl_pct:.1f}%")
    
    # ==================== CLUSTER DISTRIBUTION ====================
    st.markdown('<h2 class="sub-header">🎯 Buyer Segment Distribution</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart
        cluster_counts = filtered_df['Cluster_Name'].value_counts()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Buyer Segments Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for clusters
        fig_bar = px.bar(
            x=cluster_counts.values,
            y=cluster_counts.index,
            orientation='h',
            title="Segment Sizes",
            labels={'x': 'Number of Buyers', 'y': 'Segment'},
            color=cluster_counts.values,
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ==================== PCA VISUALIZATION ====================
    st.markdown('<h2 class="sub-header">🔍 Cluster Visualization (PCA)</h2>', unsafe_allow_html=True)
    
    with st.spinner("Generating PCA visualization..."):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features[:len(filtered_df)])
        
        fig_pca = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=filtered_df['Cluster_Name'].values,
            title="2D Projection of Buyer Segments",
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.7,
            hover_data={'Age': filtered_df['age'].values if 'age' in filtered_df.columns else None}
        )
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
    
    # ==================== SEGMENT DETAILS ====================
    st.markdown('<h2 class="sub-header">📋 Segment Profiles & Insights</h2>', unsafe_allow_html=True)
    
    tabs = st.tabs([f"{name}" for name in cluster_names.values()])
    
    for idx, (cluster_id, cluster_name) in enumerate(cluster_names.items()):
        with tabs[idx]:
            cluster_data = filtered_df[filtered_df['Cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                # Profile metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Segment Size", f"{len(cluster_data):,} buyers")
                    st.metric("Percentage", f"{len(cluster_data)/len(filtered_df)*100:.1f}%")
                
                with col2:
                    avg_age = cluster_data['age'].mean() if 'age' in cluster_data.columns else 0
                    st.metric("Average Age", f"{avg_age:.0f} years")
                    loan_rate = cluster_data['has_loan'].mean() * 100 if 'has_loan' in cluster_data.columns else 0
                    st.metric("Loan Usage", f"{loan_rate:.1f}%")
                
                with
