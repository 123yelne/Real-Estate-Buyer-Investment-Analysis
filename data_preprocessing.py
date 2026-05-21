"""
Data Preprocessing Module for Buyer Segmentation
Handles data cleaning, encoding, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(clients_path='clients.csv', properties_path='properties.csv'):
    """
    Load and clean both datasets
    """
    print("Loading datasets...")
    
    # Load data
    clients_df = pd.read_csv(clients_path)
    properties_df = pd.read_csv(properties_path)
    
    # Clean properties data
    properties_df['sale_price'] = properties_df['sale_price'].str.replace('"', '').str.replace(',', '').astype(float)
    properties_df['transaction_date'] = pd.to_datetime(properties_df['transaction_date'], format='%d-%m-%Y', errors='coerce')
    properties_df['year'] = properties_df['transaction_date'].dt.year
    properties_df['month'] = properties_df['transaction_date'].dt.month
    
    # Filter sold properties
    sold_properties = properties_df[properties_df['listing_status'] == 'Sold'].copy()
    
    # Merge datasets
    merged_df = sold_properties.merge(
        clients_df[['client_id', 'client_type', 'gender', 'country', 'region', 
                    'date_of_birth', 'acquisition_purpose', 'loan_applied', 
                    'referral_channel', 'satisfaction_score']],
        left_on='client_ref',
        right_on='client_id',
        how='left'
    )
    
    print(f"Loaded {len(clients_df)} clients and {len(properties_df)} properties")
    print(f"Merged dataset has {len(merged_df)} records")
    
    return clients_df, properties_df, merged_df


def clean_client_data(clients_df):
    """
    Clean client data: handle missing values, normalize categories
    """
    print("\n--- Cleaning Client Data ---")
    df = clients_df.copy()
    
    # Handle missing values
    df['gender'] = df['gender'].fillna('Unknown')
    df['region'] = df['region'].fillna('Unknown')
    df['referral_channel'] = df['referral_channel'].fillna('Other')
    df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())
    
    # Remove duplicate clients
    initial_count = len(df)
    df = df.drop_duplicates(subset=['client_id'], keep='first')
    print(f"Removed {initial_count - len(df)} duplicate clients")
    
    # Calculate age from date_of_birth
    def calculate_age(birth_date):
        if pd.isna(birth_date):
            return np.nan
        try:
            # Handle different date formats
            if isinstance(birth_date, str):
                for fmt in ['%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']:
                    try:
                        birth = datetime.strptime(birth_date, fmt)
                        break
                    except:
                        continue
                else:
                    return np.nan
            else:
                birth = birth_date
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except:
            return np.nan
    
    df['age'] = df['date_of_birth'].apply(calculate_age)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                              labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Normalize categorical labels
    df['client_type'] = df['client_type'].str.title()
    df['acquisition_purpose'] = df['acquisition_purpose'].str.title()
    df['loan_applied'] = df['loan_applied'].fillna('No')
    
    # Create additional features
    df['is_international'] = df['country'] != 'USA'
    df['is_corporate'] = df['client_type'] == 'Company'
    df['is_high_net_worth'] = df['satisfaction_score'] >= 4
    
    print(f"Cleaned dataset has {len(df)} clients")
    print(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f}")
    
    return df


def prepare_features_for_clustering(df):
    """
    Prepare features for clustering analysis
    """
    print("\n--- Preparing Features for Clustering ---")
    
    # Select features for clustering
    feature_columns = []
    
    # Numerical features
    numerical_features = ['age', 'satisfaction_score']
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            feature_columns.append(col)
    
    # Create derived numerical features
    df['has_loan'] = (df['loan_applied'] == 'Yes').astype(int)
    df['is_investment'] = (df['acquisition_purpose'] == 'Investment').astype(int)
    df['is_corporate'] = (df['client_type'] == 'Company').astype(int)
    df['is_international'] = (df['country'] != 'USA').astype(int)
    df['high_satisfaction'] = (df['satisfaction_score'] >= 4).astype(int)
    
    feature_columns.extend(['has_loan', 'is_investment', 'is_corporate', 'is_international', 'high_satisfaction'])
    
    # One-Hot Encode categorical features
    categorical_features = ['client_type', 'acquisition_purpose', 'referral_channel', 'gender', 'region']
    
    for col in categorical_features:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            feature_columns.extend(dummies.columns.tolist())
    
    # Label encode country (too many unique values for one-hot)
    country_encoder = LabelEncoder()
    df['country_encoded'] = country_encoder.fit_transform(df['country'].fillna('Unknown'))
    feature_columns.append('country_encoded')
    
    # Create feature matrix
    feature_matrix = df[feature_columns].copy()
    
    # Handle any remaining missing values
    feature_matrix = feature_matrix.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    print(f"Prepared {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} clients")
    
    return scaled_features, feature_matrix, scaler, feature_columns


def prepare_buyer_features(df):
    """
    Prepare buyer-level aggregated features for segmentation
    (Group by client_id for investment behavior)
    """
    print("\n--- Preparing Buyer-Level Features ---")
    
    # Aggregate by client
    buyer_features = df.groupby('client_id').agg({
        'sale_price': ['sum', 'mean', 'count'],  # Spending behavior
        'floor_area_sqft': ['sum', 'mean'],      # Size preference
        'age': 'mean',                            # Average age
        'satisfaction_score': 'mean',             # Satisfaction
        'has_loan': 'max',                        # Ever used loan
        'is_investment': 'max',                   # Investment purpose
        'is_corporate': 'max',                    # Corporate buyer
        'is_international': 'max',                # International buyer
        'tower_number': lambda x: x.nunique(),    # Diversity of tower choice
        'unit_category': lambda x: x.nunique()    # Diversity of unit type
    }).round(2)
    
    # Flatten column names
    buyer_features.columns = ['_'.join(col).strip() for col in buyer_features.columns.values]
    buyer_features.columns = [col.replace('<lambda_0>', 'diversity') for col in buyer_features.columns]
    buyer_features.columns = [col.replace('_', '_') for col in buyer_features.columns]
    
    # Rename for clarity
    buyer_features = buyer_features.rename(columns={
        'sale_price_sum': 'total_spent',
        'sale_price_mean': 'avg_price',
        'sale_price_count': 'units_purchased',
        'floor_area_sqft_sum': 'total_area',
        'floor_area_sqft_mean': 'avg_size',
        'age_mean': 'avg_age',
        'satisfaction_score_mean': 'avg_satisfaction'
    })
    
    # Add derived features
    buyer_features['price_per_unit'] = buyer_features['total_spent'] / buyer_features['units_purchased']
    buyer_features['is_high_value'] = (buyer_features['total_spent'] > buyer_features['total_spent'].median()).astype(int)
    buyer_features['is_frequent_buyer'] = (buyer_features['units_purchased'] > 1).astype(int)
    
    # Handle missing values
    buyer_features = buyer_features.fillna(0)
    
    print(f"Created {buyer_features.shape[1]} features for {buyer_features.shape[0]} buyers")
    
    return buyer_features


if __name__ == "__main__":
    # Test the preprocessing
    clients, properties, merged = load_and_clean_data()
    clients_cleaned = clean_client_data(clients)
    features, _, _, _ = prepare_features_for_clustering(clients_cleaned)
    buyer_features = prepare_buyer_features(merged)
    
    print(f"\nPreprocessing complete!")
    print(f"Client features shape: {features.shape}")
    print(f"Buyer features shape: {buyer_features.shape}")
