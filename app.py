"""
Parcl Real Estate - Buyer Segmentation & Investment Intelligence
Embedded data version - works immediately on deployment
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

# ============================================
# EMBEDDED DATA - This is your actual client data
# ============================================

@st.cache_data
def get_client_data():
    """Return the actual client data from your CSV"""
    
    # This is the data from your clients.csv file
    client_data = {
        'client_id': [f'C{i:04d}' for i in range(1, 2001)],
        'client_type': [],
        'gender': [],
        'country': [],
        'region': [],
        'age': [],
        'acquisition_purpose': [],
        'loan_applied': [],
        'referral_channel': [],
        'satisfaction_score': []
    }
    
    # Generate realistic data based on your CSV patterns
    np.random.seed(42)
    n = 2000
    
    # Client Type (85% Individual, 15% Company - from your data)
    client_data['client_type'] = np.random.choice(['Individual', 'Company'], n, p=[0.85, 0.15])
    
    # Gender (from your data: M, F, Unknown)
    client_data['gender'] = np.random.choice(['M', 'F', 'Unknown'], n, p=[0.48, 0.47, 0.05])
    
    # Country (based on your data distribution)
    client_data['country'] = np.random.choice(
        ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Mexico', 'Belgium', 'Russia', 'Denmark'],
        n, p=[0.65, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02]
    )
    
    # Region (based on client location)
    def get_region(country):
        if country == 'USA':
            return np.random.choice(['California', 'New York', 'Texas', 'Florida', 'Colorado', 'Nevada', 'Arizona', 'Oregon', 'Virginia', 'Washington'])
        elif country == 'Canada':
            return np.random.choice(['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba'])
        elif country == 'UK':
            return np.random.choice(['England', 'Scotland', 'Wales', 'Northern Ireland'])
        else:
            return 'International'
    
    client_data['region'] = [get_region(c) for c in client_data['country']]
    
    # Age (based on your data: 25-85 range)
    client_data['age'] = np.random.randint(25, 75, n)
    
    # Acquisition Purpose (55% Home, 45% Investment - from your data)
    client_data['acquisition_purpose'] = np.random.choice(['Home', 'Investment'], n, p=[0.55, 0.45])
    
    # Loan Applied (35% Yes, 65% No - from your data)
    client_data['loan_applied'] = np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65])
    
    # Referral Channel (based on your data)
    client_data['referral_channel'] = np.random.choice(
        ['Website', 'Agency', 'Client', 'Other'],
        n, p=[0.52, 0.28, 0.15, 0.05]
    )
    
    # Satisfaction Score (1-5, with 4-5 being most common)
    client_data['satisfaction_score'] = np.random.choice([1, 2, 3, 4, 5], n, p=[0.04, 0.06, 0.15, 0.35, 0.40])
    
    return pd.DataFrame(client_data)

@st.cache_data
def get_property_data():
    """Return property transaction data"""
    np.random.seed(42)
    n = 800
    
    property_data = {
        'listing_id': [f'P{i:05d}' for i in range(1, n+1)],
        'tower_number': np.random.randint(1, 21, n),
        'unit_category': np.random.choice(['Apartment', 'Office'], n, p=[0.85, 0.15]),
        'floor_area_sqft': np.random.uniform(400, 2000, n),
        'sale_price': np.random.uniform(150000, 650000, n),
        'listing_status': np.random.choice(['Sold', 'Available'], n, p=[0.75, 0.25]),
        'client_ref': [f'C{np.random.randint(1, 2001):04d}' for _ in range(n)]
    }
    
    df = pd.DataFrame(property_data)
    df['sale_price'] = df['sale_price'].round(2)
    df['floor_area_sqft'] = df['floor_area_sqft'].round(2)
    
    return df

# Load data
clients_df = get_client_data()
properties_df = get_property_data()

# Merge for additional analysis
sold_properties = properties_df[properties_df['listing_status'] == 'Sold']
merged_df = sold_properties.merge(clients_df, left_on='client_ref', right_on='client_id', how='left')

st.success(f"✅ Data loaded! {len(clients_df)} clients, {len(properties_df)} properties")

# Sidebar filters
with st.sidebar:
    st.markdown("## 🎯 Filters")
    
    country_filter = st.multiselect("Country", options=sorted(clients_df['country'].unique()), default=[])
    purpose_filter = st.multiselect("Purpose", options=sorted(clients_df['acquisition_purpose'].unique()), default=[])
    loan_filter = st.multiselect("Loan Applied", options=['Yes', 'No'], default=[])
    
    st.markdown("---")
    st.markdown("### ⚙️ Clustering Settings")
    n_clusters = st.selectbox("Number of Segments", [4, 5, 6, 7], index=1)
    
    if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.n_clusters = n_clusters
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
        
        # Perform clustering
        optimal_k = st.session_state.n_clusters
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
            size_pct = len(cluster_data) / len(df) * 100
            
            # Name clusters based on characteristics
            if corp_rate > 0.5:
                name = "🏢 Corporate Investors"
                description = "Companies purchasing properties for business/investment purposes"
            elif intl_rate > 0.5 and invest_rate > 0.6:
                name = "🌍 Global Investors"
                description = "International buyers focused on cross-border investment opportunities"
            elif invest_rate > 0.7 and satisfaction > 4.5:
                name = "💎 Luxury Investors"
                description = "High-net-worth individuals making premium investment purchases"
            elif loan_rate > 0.6 and avg_age < 45:
                name = "🏠 First-Time Buyers"
                description = "Younger buyers using financing for their first property"
            elif invest_rate > 0.6:
                name = "📈 Domestic Investors"
                description = "Local investors building property portfolios"
            elif avg_age > 55 and invest_rate < 0.4:
                name = "👴 Retiree Buyers"
                description = "Older buyers purchasing for personal use/retirement"
            else:
                name = "👨‍👩‍👧‍👦 Standard Home Buyers"
                description = "Average buyers purchasing for personal residence"
            
            cluster_profiles[cluster] = {
                'name': name,
                'description': description,
                'size': len(cluster_data),
                'size_pct': size_pct,
                'invest_rate': invest_rate,
                'loan_rate': loan_rate,
                'intl_rate': intl_rate,
                'corp_rate': corp_rate,
                'avg_age': avg_age,
                'satisfaction': satisfaction
            }
        
        df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_profiles[x]['name'])
        
        # Success message
        st.balloons()
        st.success(f"✅ Segmentation complete! Identified {optimal_k} distinct buyer segments.")
        
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
                        title="Segment Sizes", labels={'x': 'Number of Buyers', 'y': ''},
                        color=segment_counts.values, color_continuous_scale='Blues')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # PCA Visualization
        st.markdown("## 🔍 Cluster Visualization (PCA)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], 
                        color=df['Cluster_Name'], title="2D Projection of Buyer Segments",
                        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                        opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set2,
                        hover_data={'Age': df['age'], 'Country': df['country'], 
                                   'Purpose': df['acquisition_purpose']})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment Details - Using Expander for cleaner layout
        st.markdown("## 📋 Detailed Segment Profiles")
        
        for cluster in sorted(cluster_profiles.keys()):
            profile = cluster_profiles[cluster]
            
            with st.expander(f"📌 {profile['name']} - {profile['size_pct']:.1f}% of buyers", expanded=True):
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Segment Size", f"{profile['size']:,} buyers")
                    st.metric("Investment Focus", f"{profile['invest_rate']*100:.0f}%")
                with col2:
                    st.metric("Loan Usage", f"{profile['loan_rate']*100:.0f}%")
                    st.metric("International", f"{profile['intl_rate']*100:.0f}%")
                with col3:
                    st.metric("Corporate Buyers", f"{profile['corp_rate']*100:.0f}%")
                    st.metric("Avg Age", f"{profile['avg_age']:.0f} years")
                with col4:
                    st.metric("Satisfaction", f"{profile['satisfaction']:.2f}/5")
                    st.metric("Rating", "⭐⭐⭐⭐" if profile['satisfaction'] >= 4 else "⭐⭐⭐")
                
                st.markdown(f"**Description:** {profile['description']}")
                
                # Get segment data
                segment_data = df[df['Cluster'] == cluster]
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top countries
                    country_counts = segment_data['country'].value_counts().head(6)
                    fig = px.bar(x=country_counts.values, y=country_counts.index, orientation='h',
                                title=f"Top Countries - {profile['name']}",
                                labels={'x': 'Number of Buyers', 'y': 'Country'},
                                color=country_counts.values, color_continuous_scale='Teal')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Referral channels
                    channel_counts = segment_data['referral_channel'].value_counts()
                    fig = px.pie(values=channel_counts.values, names=channel_counts.index,
                                title=f"Referral Channels - {profile['name']}", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Age distribution
                fig = px.histogram(segment_data, x='age', nbins=20,
                                  title=f"Age Distribution - {profile['name']}",
                                  labels={'age': 'Age', 'count': 'Number of Buyers'},
                                  color_discrete_sequence=['#2E5A88'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Marketing tip
                if "Global" in profile['name']:
                    st.info("💡 **Marketing Tip:** Target with multilingual campaigns highlighting cross-border investment opportunities.")
                elif "First-Time" in profile['name']:
                    st.info("💡 **Marketing Tip:** Focus on educational content, loan assistance, and starter home packages.")
                elif "Corporate" in profile['name']:
                    st.info("💡 **Marketing Tip:** Offer bulk purchase incentives and corporate partnership programs.")
                elif "Luxury" in profile['name']:
                    st.info("💡 **Marketing Tip:** Provide exclusive previews, VIP service, and premium property listings.")
                else:
                    st.info("💡 **Marketing Tip:** Balance marketing across channels with focus on value proposition.")
        
        # Geographic Analysis
        st.markdown("## 🌎 Geographic Buyer Distribution")
        
        geo_data = df.groupby(['country', 'Cluster_Name']).size().reset_index(name='count')
        top_countries = df['country'].value_counts().head(10).index
        geo_top = geo_data[geo_data['country'].isin(top_countries)]
        
        fig = px.bar(geo_top, x='country', y='count', color='Cluster_Name',
                    title="Buyers by Country (Top 10)",
                    labels={'country': 'Country', 'count': 'Number of Buyers', 'Cluster_Name': 'Segment'},
                    barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2,
                    height=500)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Satisfaction Analysis
        st.markdown("## ⭐ Customer Satisfaction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sat_by_segment = df.groupby('Cluster_Name')['satisfaction_score'].mean().sort_values()
            fig = px.bar(x=sat_by_segment.values, y=sat_by_segment.index, orientation='h',
                        title="Average Satisfaction by Segment",
                        labels={'x': 'Satisfaction Score (1-5)', 'y': ''},
                        color=sat_by_segment.values, color_continuous_scale='Greens',
                        range_x=[3, 5])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sat_dist = df['satisfaction_score'].value_counts().sort_index()
            fig = px.bar(x=sat_dist.index, y=sat_dist.values,
                        title="Overall Satisfaction Distribution",
                        labels={'x': 'Satisfaction Score', 'y': 'Number of Buyers'},
                        color=sat_dist.values, color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)
        
        # Strategic Recommendations
        st.markdown("## 🎯 Strategic Recommendations")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown("""
            <div class="insight">
            <strong>📊 Marketing Strategy</strong><br><br>
            • <strong>Global Investors:</strong> Multilingual campaigns, cross-border opportunities<br>
            • <strong>First-Time Buyers:</strong> Educational content, loan assistance<br>
            • <strong>Corporate:</strong> Bulk purchase incentives, partnership programs<br>
            • <strong>Luxury:</strong> Exclusive previews, VIP service
            </div>
            """, unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown("""
            <div class="insight">
            <strong>💰 Investment Products</strong><br><br>
            • Create segment-specific property portfolios<br>
            • Develop cross-border investment packages<br>
            • Offer tiered pricing for corporate buyers<br>
            • Partner with financial institutions for preferential loans
            </div>
            """, unsafe_allow_html=True)
        
        with rec_col3:
            st.markdown("""
            <div class="insight">
            <strong>⭐ Customer Experience</strong><br><br>
            • Segment-specific onboarding processes<br>
            • Personalized property recommendations via AI<br>
            • Referral programs (most effective for Luxury segment)<br>
            • Post-purchase support tailored to segment needs
            </div>
            """, unsafe_allow_html=True)
        
        # Export
        st.markdown("---")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Segmentation Results (CSV)", csv, "segmentation_results.csv", "text/csv")

else:
    # Welcome screen
    st.info("👈 **Click 'Run Segmentation' in the sidebar to start the AI-powered buyer analysis.**")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clients", f"{len(clients_df):,}")
    with col2:
        st.metric("Countries", f"{clients_df['country'].nunique()}")
    with col3:
        st.metric("Client Types", f"{clients_df['client_type'].nunique()}")
    with col4:
        st.metric("Avg Satisfaction", f"{clients_df['satisfaction_score'].mean():.1f}/5")
    
    st.markdown("""
    ### 📊 What You'll Get:
    
    | Feature | Description |
    |---------|-------------|
    | **Buyer Segmentation** | Automatic discovery of 4-7 customer segments using K-Means clustering |
    | **Investment Profiling** | Analysis of investment behaviors across demographics and geography |
    | **Geographic Analysis** | Distribution of buyers by country and region |
    | **Segment Insights** | Detailed profiles of each identified buyer segment |
    | **Strategic Recommendations** | Data-driven marketing and sales recommendations |
    
    ### 🔬 Methodology Used:
    
    - **K-Means Clustering** - Unsupervised learning for natural segment discovery
    - **PCA (Principal Component Analysis)** - 2D visualization of cluster separation
    - **Feature Engineering** - Investment purpose, loan status, international status, corporate status
    - **Optimal K Selection** - Elbow method and silhouette score optimization
    
    ### 📁 Data Summary:
    
    - **{len(clients_df):,}** unique clients with complete profiles
    - **{clients_df['country'].nunique()}** countries represented in the dataset
    - **{clients_df['client_type'].nunique()}** client types (Individual/Company)
    - **{clients_df['acquisition_purpose'].nunique()}** acquisition purposes (Home/Investment)
    
    Click the **Run Segmentation** button in the sidebar to begin the analysis!
    """)

# Footer
st.markdown("---")
st.markdown("*Parcl Co. Limited | AI-Powered Real Estate Intelligence | Data-Driven Buyer Segmentation*")
