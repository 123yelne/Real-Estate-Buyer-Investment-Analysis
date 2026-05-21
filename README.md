# Real-Estate-Buyer-Investment-Analysis
🏠 Parcl Real Estate - Buyer Segmentation & Investment Intelligence
Overview
AI-powered buyer segmentation system that identifies distinct customer groups using machine learning clustering, enabling data-driven marketing and sales strategies.

Key Features
5 Buyer Segments automatically identified (Global Investors, First-Time Buyers, Corporate Investors, Luxury Investors, Standard Home Buyers)

Interactive Dashboard with real-time filtering by country, purpose, and age

Investment Pattern Analysis across demographics and geography

Strategic Recommendations tailored to each segment

Tech Stack
Python 3.9+ | scikit-learn | Streamlit | Plotly | Pandas

Quick Start
bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# Or run complete analysis
python run_project.py
Project Structure
text
├── app.py                    # Streamlit dashboard
├── clustering_analysis.py    # K-Means & Hierarchical clustering
├── data_preprocessing.py     # Data cleaning & feature engineering
├── requirements.txt          # Dependencies
├── research_paper.md         # Full methodology & findings
└── run_project.py           # Execution script
Dashboard Features
📊 Segment distribution (pie/bar charts)

🔍 PCA visualization of clusters

📋 Detailed segment profiles

🌎 Geographic buyer heatmaps

⭐ Satisfaction analysis

🎯 Strategic recommendations

Data Requirements
clients.csv - Client demographics (client_id, age, country, acquisition_purpose, etc.)

properties.csv - Property transactions (sale_price, unit_type, status, etc.)

Key Insights
5 distinct segments with Silhouette Score 0.45

70% of international buyers are Global Investors

65% loan utilization among First-Time Buyers

4.8/5 satisfaction for Luxury Investors

Outputs
segmentation_results.csv - Clustered client data

Live Streamlit dashboard (port 8501)

Research paper with actionable recommendations
