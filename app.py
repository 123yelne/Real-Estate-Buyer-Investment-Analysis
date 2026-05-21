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
                
                with col3:
                    invest_rate = cluster_data['is_investment'].mean() * 100 if 'is_investment' in cluster_data.columns else 0
                    st.metric("Investment Focus", f"{invest_rate:.1f}%")
                    corporate_rate = cluster_data['is_corporate'].mean() * 100 if 'is_corporate' in cluster_data.columns else 0
                    st.metric("Corporate Buyers", f"{corporate_rate:.1f}%")
                
                with col4:
                    satisfaction = cluster_data['satisfaction_score'].mean() if 'satisfaction_score' in cluster_data.columns else 0
                    st.metric("Satisfaction", f"{satisfaction:.2f}/5")
                    intl_rate = cluster_data['is_international'].mean() * 100 if 'is_international' in cluster_data.columns else 0
                    st.metric("International", f"{intl_rate:.1f}%")
                
                # Demographics charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age distribution
                    if 'age' in cluster_data.columns and not cluster_data['age'].isna().all():
                        fig_age = px.histogram(
                            cluster_data, x='age', nbins=20,
                            title=f'Age Distribution - {cluster_name}',
                            labels={'age': 'Age', 'count': 'Number of Buyers'},
                            color_discrete_sequence=['#2E5A88']
                        )
                        st.plotly_chart(fig_age, use_container_width=True)
                
                with col2:
                    # Purpose distribution
                    if 'acquisition_purpose' in cluster_data.columns:
                        purpose_counts = cluster_data['acquisition_purpose'].value_counts()
                        fig_purpose = px.pie(
                            values=purpose_counts.values,
                            names=purpose_counts.index,
                            title=f'Acquisition Purpose - {cluster_name}',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_purpose, use_container_width=True)
                
                # Geographic distribution
                st.markdown("#### 🌍 Geographic Distribution")
                if 'country' in cluster_data.columns:
                    country_counts = cluster_data['country'].value_counts().head(10)
                    fig_country = px.bar(
                        x=country_counts.values,
                        y=country_counts.index,
                        orientation='h',
                        title=f'Top 10 Countries - {cluster_name}',
                        labels={'x': 'Number of Buyers', 'y': 'Country'},
                        color=country_counts.values,
                        color_continuous_scale='Teal'
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
                
                # Referral channels
                st.markdown("#### 📢 Referral Channels")
                if 'referral_channel' in cluster_data.columns:
                    channel_counts = cluster_data['referral_channel'].value_counts()
                    fig_channel = px.bar(
                        x=channel_counts.index,
                        y=channel_counts.values,
                        title=f'Referral Channels - {cluster_name}',
                        labels={'x': 'Channel', 'y': 'Number of Buyers'},
                        color=channel_counts.values,
                        color_continuous_scale='Orange'
                    )
                    st.plotly_chart(fig_channel, use_container_width=True)
                
                # Insight box
                st.markdown(f"""
                <div class="insight-box">
                    <strong>💡 Segment Insights:</strong><br>
                    • This segment represents {len(cluster_data)/len(filtered_df)*100:.1f}% of all buyers<br>
                    • {invest_rate:.0f}% purchase for investment purposes<br>
                    • {loan_rate:.0f}% utilize financing options<br>
                    • {intl_rate:.0f}% are international buyers<br>
                    • Average satisfaction rating: {satisfaction:.1f}/5
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No buyers in this segment with current filters.")
    
    # ==================== INVESTMENT PATTERNS ====================
    st.markdown('<h2 class="sub-header">💰 Investment Behavior Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Investment vs Non-Investment by cluster
        if 'is_investment' in filtered_df.columns and 'Cluster_Name' in filtered_df.columns:
            invest_by_cluster = filtered_df.groupby(['Cluster_Name', 'is_investment']).size().unstack(fill_value=0)
            invest_by_cluster_pct = invest_by_cluster.div(invest_by_cluster.sum(axis=1), axis=0) * 100
            
            fig_invest = px.bar(
                invest_by_cluster_pct,
                x=invest_by_cluster_pct.index,
                y=[0, 1],
                title="Investment vs Personal Use by Segment",
                labels={'value': 'Percentage', 'variable': 'Purpose', 'index': 'Segment'},
                barmode='stack',
                color_discrete_map={0: '#A8D5E5', 1: '#1E3A5F'}
            )
            fig_invest.update_layout(legend=dict(title='', labels={0: 'Personal', 1: 'Investment'}))
            st.plotly_chart(fig_invest, use_container_width=True)
    
    with col2:
        # Loan usage by cluster
        if 'has_loan' in filtered_df.columns and 'Cluster_Name' in filtered_df.columns:
            loan_by_cluster = filtered_df.groupby(['Cluster_Name', 'has_loan']).size().unstack(fill_value=0)
            loan_by_cluster_pct = loan_by_cluster.div(loan_by_cluster.sum(axis=1), axis=0) * 100
            
            fig_loan = px.bar(
                loan_by_cluster_pct,
                x=loan_by_cluster_pct.index,
                y=[0, 1],
                title="Loan Usage by Segment",
                labels={'value': 'Percentage', 'variable': 'Loan', 'index': 'Segment'},
                barmode='stack',
                color_discrete_map={0: '#FFB74D', 1: '#E65100'}
            )
            fig_loan.update_layout(legend=dict(title='', labels={0: 'No Loan', 1: 'Loan Applied'}))
            st.plotly_chart(fig_loan, use_container_width=True)
    
    # ==================== GEOGRAPHIC HEATMAP ====================
    st.markdown('<h2 class="sub-header">🌎 Geographic Buyer Distribution</h2>', unsafe_allow_html=True)
    
    if 'country' in filtered_df.columns and 'Cluster_Name' in filtered_df.columns:
        geo_data = filtered_df.groupby(['country', 'Cluster_Name']).size().reset_index(name='count')
        
        # Top 15 countries
        top_countries = filtered_df['country'].value_counts().head(15).index
        geo_top = geo_data[geo_data['country'].isin(top_countries)]
        
        fig_geo = px.bar(
            geo_top,
            x='country',
            y='count',
            color='Cluster_Name',
            title="Buyer Distribution by Country",
            labels={'country': 'Country', 'count': 'Number of Buyers', 'Cluster_Name': 'Segment'},
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_geo.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_geo, use_container_width=True)
    
    # ==================== SATISFACTION ANALYSIS ====================
    st.markdown('<h2 class="sub-header">⭐ Customer Satisfaction Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'satisfaction_score' in filtered_df.columns and 'Cluster_Name' in filtered_df.columns:
            sat_by_cluster = filtered_df.groupby('Cluster_Name')['satisfaction_score'].mean().sort_values()
            fig_sat = px.bar(
                x=sat_by_cluster.values,
                y=sat_by_cluster.index,
                orientation='h',
                title="Average Satisfaction by Segment",
                labels={'x': 'Satisfaction Score (1-5)', 'y': 'Segment'},
                color=sat_by_cluster.values,
                color_continuous_scale='Greens',
                range_x=[0, 5]
            )
            st.plotly_chart(fig_sat, use_container_width=True)
    
    with col2:
        if 'satisfaction_score' in filtered_df.columns:
            fig_sat_dist = px.histogram(
                filtered_df, x='satisfaction_score', nbins=5,
                title="Overall Satisfaction Distribution",
                labels={'satisfaction_score': 'Satisfaction Score', 'count': 'Number of Buyers'},
                color_discrete_sequence=['#2E7D32']
            )
            fig_sat_dist.update_layout(bargap=0.1)
            st.plotly_chart(fig_sat_dist, use_container_width=True)
    
    # ==================== RECOMMENDATIONS ====================
    st.markdown('<h2 class="sub-header">🎯 Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <strong>📊 Marketing Strategy</strong><br><br>
            • Target Global Investors with multilingual campaigns<br>
            • Focus Corporate segment with bulk purchase incentives<br>
            • Engage First-Time Buyers with educational content<br>
            • Use high-end channels for Luxury Investors
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <strong>💰 Investment Opportunities</strong><br><br>
            • Develop investor-specific property portfolios<br>
            • Create cross-border investment packages<br>
            • Offer tiered pricing for corporate buyers<br>
            • Provide financing partnerships for loan users
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
            <strong>⭐ Customer Experience</strong><br><br>
            • Maintain high satisfaction (currently 4+/5)<br>
            • Implement segment-specific loyalty programs<br>
            • Enhance referral program effectiveness<br>
            • Provide personalized property recommendations
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen before clustering
    st.info("👈 Click **Run Segmentation Analysis** in the sidebar to start the AI-powered buyer segmentation.")
    
    st.markdown("""
    ### 📊 What You'll Get:
    
    | Feature | Description |
    |---------|-------------|
    | **Buyer Segmentation** | Automatic discovery of customer segments using K-Means clustering |
    | **Investment Profiling** | Analysis of investment behaviors across demographics |
    | **Geographic Analysis** | Distribution of buyers by country and region |
    | **Segment Insights** | Detailed profiles of each identified buyer segment |
    | **Strategic Recommendations** | Data-driven marketing and sales recommendations |
    
    ### 🔬 Methodology:
    
    The analysis uses:
    - **K-Means Clustering** for buyer segmentation
    - **PCA** for visualization of cluster separation
    - **Feature Engineering** including loan status, investment purpose, international status
    - **Silhouette Score** optimization for cluster quality
    
    ### 📁 Data Overview:
    
    - **{len(clients_cleaned)}** unique clients
    - **{len(merged)}** property transactions
    - **{clients_cleaned['country'].nunique()}** countries represented
    - **{clients_cleaned['client_type'].nunique()}** client types
    
    Click the button above to begin the analysis!
    """.format(
        clients_cleaned=clients_cleaned,
        merged=merged
    ), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Parcl Co. Limited | AI-Powered Real Estate Intelligence | Data-Driven Buyer Segmentation</p>",
    unsafe_allow_html=True
)
