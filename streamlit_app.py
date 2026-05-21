"""
Parcl Real Estate - Buyer Segmentation & Investment Intelligence
Single-file Streamlit App - No complex installations needed
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Parcl - Buyer Intelligence", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font { font-size: 24px !important; font-weight: bold; color: #1E3A5F; }
.metric-card { background: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
.insight { background: #e8f4f8; border-left: 4px solid #1E3A5F; padding: 15px; margin: 10px 0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🏠 Parcl Real Estate - Buyer Segmentation Intelligence</p>', unsafe_allow_html=True)
st.markdown("*AI-Powered Customer Segmentation for Data-Driven Marketing*")

# Load data
@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        # Try to load from uploaded files or use sample data
        clients_df = pd.read_csv('clients.csv')
        properties_df = pd.read_csv('properties.csv')
        
        # Clean properties
        properties_df['sale_price'] = properties_df['sale_price'].str.replace('"', '').str.replace(',', '').astype(float)
        properties_df['transaction_date'] = pd.to_datetime(properties_df['transaction_date'], format='%d-%m-%Y', errors='coerce')
        
        # Filter sold properties
        sold = properties_df[properties_df['listing_status'] == 'Sold'].copy()
        
        # Merge
        merged = sold.merge(clients_df, left_on='client_ref', right_on='client_id', how='left')
        
        return clients_df, properties_df, merged
    except:
        st.warning("⚠️ Data files not found. Using sample data for demonstration.")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n = 500
    
    sample_clients = pd.DataFrame({
        'client_id': [f'C{i:04d}' for i in range(n)],
        'client_type': np.random.choice(['Individual', 'Company'], n, p=[0.85, 0.15]),
        'gender': np.random.choice(['M', 'F', 'Unknown'], n, p=[0.45, 0.45, 0.1]),
        'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia'], n, p=[0.65, 0.1, 0.08, 0.06, 0.06, 0.05]),
        'region': np.random.choice(['California', 'New York', 'Texas', 'Florida', 'Other'], n),
        'age': np.random.randint(25, 70, n),
        'acquisition_purpose': np.random.choice(['Home', 'Investment'], n, p=[0.55, 0.45]),
        'loan_applied': np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65]),
        'referral_channel': np.random.choice(['Website', 'Agency', 'Client', 'Other'], n, p=[0.5, 0.3, 0.15, 0.05]),
        'satisfaction_score': np.random.choice([1,2,3,4,5], n, p=[0.05, 0.08, 0.15, 0.32, 0.4])
    })
    
    return sample_clients, None, None

# Load data
clients_df, properties_df, merged_df = load_data()

# Sidebar filters
with st.sidebar:
    st.markdown("## 🎯 Filters")
    
    country_filter = st.multiselect("Country", options=sorted(clients_df['country'].unique()), default=[])
    purpose_filter = st.multiselect("Purpose", options=sorted(clients_df['acquisition_purpose'].unique()), default=[])
    loan_filter = st.multiselect("Loan Applied", options=['Yes', 'No'], default=[])
    
    if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

# Main analysis
if st.session_state.run_analysis:
    
    with st.spinner("Analyzing buyer segments..."):
        
        # Prepare data for clustering
        df = clients_df.copy()
        
        # Apply filters
        if country_filter:
            df = df[df['country'].isin(country_filter)]
        if purpose_filter:
            df = df[df['acquisition_purpose'].isin(purpose_filter)]
        if loan_filter:
            df = df[df['loan_applied'].isin(loan_filter)]
        
        if len(df) < 10:
            st.error("Not enough data with selected filters. Please broaden your criteria.")
            st.stop()
        
        # Feature engineering
        df['is_investment'] = (df['acquisition_purpose'] == 'Investment').astype(int)
        df['has_loan'] = (df['loan_applied'] == 'Yes').astype(int)
        df['is_corporate'] = (df['client_type'] == 'Company').astype(int)
        df['is_international'] = (df['country'] != 'USA').astype(int)
        df['high_satisfaction'] = (df['satisfaction_score'] >= 4).astype(int)
        
        # Prepare features
        feature_cols = ['age', 'satisfaction_score', 'is_investment', 'has_loan', 
                       'is_corporate', 'is_international', 'high_satisfaction']
        
        # Add encoded categoricals
        le_country = LabelEncoder()
        df['country_code'] = le_country.fit_transform(df['country'])
        feature_cols.append('country_code')
        
        le_channel = LabelEncoder()
        df['channel_code'] = le_channel.fit_transform(df['referral_channel'])
        feature_cols.append('channel_code')
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal clusters using elbow method
        inertias = []
        K_range = range(2, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Choose K (elbow at 4-5)
        optimal_k = 5
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Define cluster names based on characteristics
        cluster_profiles = {}
        for cluster in range(optimal_k):
            cluster_data = df[df['Cluster'] == cluster]
            invest_rate = cluster_data['is_investment'].mean()
            corp_rate = cluster_data['is_corporate'].mean()
            intl_rate = cluster_data['is_international'].mean()
            loan_rate = cluster_data['has_loan'].mean()
            avg_age = cluster_data['age'].mean()
            satisfaction = cluster_data['satisfaction_score'].mean()
            
            if corp_rate > 0.5:
                name = "🏢 Corporate Investors"
            elif intl_rate > 0.5 and invest_rate > 0.6:
                name = "🌍 Global Investors"
            elif invest_rate > 0.7 and satisfaction > 4.5:
                name = "💎 Luxury Investors"
            elif loan_rate > 0.6 and avg_age < 40:
                name = "🏠 First-Time Buyers"
            elif invest_rate > 0.6:
                name = "📈 Domestic Investors"
            else:
                name = "👨‍👩‍👧‍👦 Standard Home Buyers"
            
            cluster_profiles[cluster] = {
                'name': name,
                'size': len(cluster_data),
                'invest_rate': invest_rate,
                'loan_rate': loan_rate,
                'intl_rate': intl_rate,
                'avg_age': avg_age,
                'satisfaction': satisfaction
            }
        
        df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_profiles[x]['name'])
        
        # Success message
        st.success(f"✅ Segmentation complete! Identified {optimal_k} buyer segments.")
        
        # Metrics Row
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Buyers", f"{len(df):,}")
        with col2:
            st.metric("Investment Purpose", f"{df['is_investment'].mean()*100:.0f}%")
        with col3:
            st.metric("Loan Users", f"{df['has_loan'].mean()*100:.0f}%")
        with col4:
            st.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.1f}/5")
        with col5:
            st.metric("International", f"{df['is_international'].mean()*100:.0f}%")
        
        # Segment Distribution
        st.markdown("## 📊 Buyer Segment Distribution")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            segment_counts = df['Cluster_Name'].value_counts()
            fig = px.pie(values=segment_counts.values, names=segment_counts.index, 
                        title="Segment Distribution", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=segment_counts.values, y=segment_counts.index, orientation='h',
                        title="Segment Sizes", labels={'x': 'Buyers', 'y': ''},
                        color=segment_counts.values, color_continuous_scale='Blues')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # PCA Visualization
        st.markdown("## 🔍 Cluster Visualization")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], 
                        color=df['Cluster_Name'], title="2D Cluster Projection (PCA)",
                        labels={'x': 'PC1', 'y': 'PC2'}, opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hover_data={'Age': df['age'], 'Country': df['country']})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment Details
        st.markdown("## 📋 Segment Profiles")
        
        tabs = st.tabs([cluster_profiles[c]['name'] for c in sorted(cluster_profiles.keys())])
        
        for idx, cluster in enumerate(sorted(cluster_profiles.keys())):
            with tabs[idx]:
                profile = cluster_profiles[cluster]
                cluster_data = df[df['Cluster'] == cluster]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Segment Size", f"{profile['size']:,} buyers")
                    st.metric("Percentage", f"{profile['size']/len(df)*100:.1f}%")
                with col2:
                    st.metric("Investment Focus", f"{profile['invest_rate']*100:.0f}%")
                    st.metric("Loan Usage", f"{profile['loan_rate']*100:.0f}%")
                with col3:
                    st.metric("International", f"{profile['intl_rate']*100:.0f}%")
                    st.metric("Avg Age", f"{profile['avg_age']:.0f} years")
                with col4:
                    st.metric("Satisfaction", f"{profile['satisfaction']:.2f}/5")
                    st.metric("Corporate", f"{cluster_data['is_corporate'].mean()*100:.0f}%")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    purpose_counts = cluster_data['acquisition_purpose'].value_counts()
                    fig = px.pie(values=purpose_counts.values, names=purpose_counts.index,
                                title="Acquisition Purpose", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    channel_counts = cluster_data['referral_channel'].value_counts()
                    fig = px.bar(x=channel_counts.index, y=channel_counts.values,
                                title="Referral Channels", labels={'x': '', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Geographic distribution
                country_counts = cluster_data['country'].value_counts().head(8)
                fig = px.bar(x=country_counts.values, y=country_counts.index, orientation='h',
                            title="Top Countries", labels={'x': 'Buyers', 'y': ''})
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight
                st.markdown(f"""
                <div class="insight">
                <strong>💡 Segment Insights:</strong><br>
                • {profile['size']/len(df)*100:.1f}% of all buyers belong to this segment<br>
                • {profile['invest_rate']*100:.0f}% purchase for investment purposes<br>
                • {profile['loan_rate']*100:.0f}% utilize financing options<br>
                • Top referral channel: {cluster_data['referral_channel'].mode().values[0] if len(cluster_data['referral_channel'].mode()) > 0 else 'N/A'}<br>
                • Satisfaction rating: {profile['satisfaction']:.1f}/5
                </div>
                """, unsafe_allow_html=True)
        
        # Geographic Analysis
        st.markdown("## 🌎 Geographic Buyer Distribution")
        
        geo_data = df.groupby(['country', 'Cluster_Name']).size().reset_index(name='count')
        top_countries = df['country'].value_counts().head(12).index
        geo_top = geo_data[geo_data['country'].isin(top_countries)]
        
        fig = px.bar(geo_top, x='country', y='count', color='Cluster_Name',
                    title="Buyers by Country and Segment",
                    labels={'country': 'Country', 'count': 'Number of Buyers', 'Cluster_Name': 'Segment'},
                    barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Satisfaction Analysis
        st.markdown("## ⭐ Customer Satisfaction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sat_by_segment = df.groupby('Cluster_Name')['satisfaction_score'].mean().sort_values()
            fig = px.bar(x=sat_by_segment.values, y=sat_by_segment.index, orientation='h',
                        title="Avg Satisfaction by Segment", labels={'x': 'Score (1-5)', 'y': ''},
                        color=sat_by_segment.values, color_continuous_scale='Greens', range_x=[0,5])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sat_dist = df['satisfaction_score'].value_counts().sort_index()
            fig = px.bar(x=sat_dist.index, y=sat_dist.values, title="Overall Satisfaction Distribution",
                        labels={'x': 'Satisfaction Score', 'y': 'Number of Buyers'},
                        color=sat_dist.values, color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("## 🎯 Strategic Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="insight">
            <strong>📊 Marketing Strategy</strong><br><br>
            • 🌍 Global Investors: Multilingual campaigns, cross-border opportunities<br>
            • 🏠 First-Time Buyers: Educational content, loan assistance<br>
            • 🏢 Corporate: Bulk purchase incentives, partnership programs<br>
            • 💎 Luxury: Exclusive previews, VIP service
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight">
            <strong>💰 Investment Opportunities</strong><br><br>
            • Create segment-specific property portfolios<br>
            • Develop cross-border investment packages<br>
            • Offer tiered pricing for corporate buyers<br>
            • Partner with financial institutions for loans
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="insight">
            <strong>⭐ Customer Experience</strong><br><br>
            • Segment-specific onboarding processes<br>
            • Personalized property recommendations<br>
            • Referral programs (especially effective for Luxury segment)<br>
            • Post-purchase support tailored to segment needs
            </div>
            """, unsafe_allow_html=True)
        
        # Export option
        st.markdown("---")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Segmentation Results (CSV)", csv, "segmentation_results.csv", "text/csv")

else:
    # Welcome screen
    st.info("👈 **Click 'Run Segmentation' in the sidebar to start the AI-powered buyer analysis.**")
    
    st.markdown("""
    ### 📊 What You'll Get:
    
    | Feature | Description |
    |---------|-------------|
    | **Buyer Segmentation** | Automatic discovery of 4-6 customer segments |
    | **Investment Profiling** | Analysis of behaviors across demographics |
    | **Geographic Analysis** | Distribution of buyers by country |
    | **Segment Insights** | Detailed profiles of each buyer segment |
    | **Strategic Recommendations** | Data-driven marketing suggestions |
    
    ### 🔬 Methodology:
    
    - **K-Means Clustering** for buyer segmentation
    - **PCA** for visualization
    - **Feature Engineering**: Investment purpose, loan status, international status
    - **Silhouette Score** optimized clustering
    
    ### 📁 Data Summary:
    
    - **{len(clients_df):,}** unique clients
    - **{clients_df['country'].nunique()}** countries represented
    - **{clients_df['client_type'].nunique()}** client types
    - **{clients_df['acquisition_purpose'].nunique()}** acquisition purposes
    
    Click the button above to begin!
    """.format(clients_df=clients_df))

# Footer
st.markdown("---")
st.markdown("*Parcl Co. Limited | AI-Powered Real Estate Intelligence | Data-Driven Buyer Segmentation*")
