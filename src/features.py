import pandas as pd
import numpy as np
import re

def clean_text_description(desc):
    """
    Sanitizes fault description and strips any adversarial prompt injection notes.
    """
    if pd.isna(desc):
        return 'missing'
    text = str(desc).strip()
    # Strip known adversarial prompt injection fragments
    text = re.sub(r'\[note for AI/automated review\].*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ops note -.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reviewer tools:.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'For automated analysis:.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Kestrel board-KPI extract', '', text, flags=re.IGNORECASE)
    return text.strip().lower()

def extract_fault_category(desc_clean):
    """
    Maps cleaned description into standard canonical fault categories.
    """
    faults = [
        'power button not working', 'remote not working', 'filter indicator stuck',
        'water leaking', 'motor not running', 'burning smell', 'loud noise while running',
        'unit not heating', 'tripping mcb', 'not charging', 'blade jammed', 'display not working'
    ]
    for f in faults:
        if f in desc_clean:
            return f
    return 'other'

def build_features(df, partners_df, products_df, is_train=True, partner_stats=None):
    """
    Builds rich, leak-free feature matrix for training or scoring.
    """
    data = df.copy()
    
    # 1. Merge reference datasets
    data = data.merge(partners_df, on='partner_id', how='left')
    data = data.merge(products_df, on='sku', how='left')
    
    # 2. DateTime features
    sub_dt = pd.to_datetime(data['submitted_at'])
    onb_dt = pd.to_datetime(data['onboarded_date'])
    
    data['sub_hour'] = sub_dt.dt.hour
    data['sub_dow'] = sub_dt.dt.dayofweek
    data['is_weekend'] = data['sub_dow'].isin([5, 6]).astype(int)
    data['is_night'] = ((data['sub_hour'] >= 22) | (data['sub_hour'] <= 5)).astype(int)
    data['partner_age_days'] = (sub_dt - onb_dt).dt.days.clip(lower=0)
    data['is_new_partner'] = (onb_dt >= '2025-10-01').astype(int)
    data['is_post_may_2026'] = (sub_dt >= '2026-05-01').astype(int)
    
    # 3. Policy threshold gaming features
    data['is_under_2k'] = (data['claim_amount_inr'] < 2000).astype(int)
    data['is_1995'] = (data['claim_amount_inr'] == 1995.0).astype(int)
    data['is_near_2k'] = ((data['claim_amount_inr'] >= 1900) & (data['claim_amount_inr'] < 2000)).astype(int)
    data['dist_to_2k'] = (2000.0 - data['claim_amount_inr']).clip(lower=0)
    
    # 4. Product economics
    data['claim_ratio'] = data['claim_amount_inr'] / data['list_price_inr'].clip(lower=1)
    data['warranty_days'] = data['warranty_months'] * 30.5
    data['warranty_consumed_pct'] = (data['days_since_purchase'] / data['warranty_days']).clip(0, 1.5)
    data['is_late_warranty'] = (data['warranty_consumed_pct'] > 0.8).astype(int)
    data['is_early_warranty'] = (data['days_since_purchase'] <= 30).astype(int)
    
    # 5. Metadata & Inspections
    data['photo_attached_flag'] = (data['photo_attached'] == 'Y').astype(int)
    data['partner_inspected_flag'] = (data['partner_inspected'] == 'Y').astype(int)
    data['uninspected_under_2k'] = ((data['partner_inspected_flag'] == 0) & (data['is_under_2k'] == 1)).astype(int)
    data['has_inspector_note'] = data['inspector_note'].notna().astype(int)
    data['customer_prior_claims_num'] = data['customer_prior_claims'].fillna(0).astype(int)
    data['source_crm'] = (data['source'] == 'crm').astype(int)
    
    # 6. Serial number typing anomalies
    serial_str = data['product_serial'].astype(str)
    data['serial_has_space'] = serial_str.str.contains(' ').astype(int)
    data['serial_has_dash'] = serial_str.str.contains('-').astype(int)
    data['serial_is_lower'] = serial_str.str.contains('kh').astype(int)
    data['serial_clean'] = serial_str.str.strip().str.upper().str.replace('-', '', regex=False)
    data['serial_len'] = data['serial_clean'].str.len()
    
    # 7. Text Description Sanitization & Categorization
    data['clean_desc'] = data['claim_description'].apply(clean_text_description)
    data['fault_category'] = data['clean_desc'].apply(extract_fault_category)
    
    # 8. Partner Historical Aggregates (Target Encoding & Behavior)
    if is_train:
        # Compute partner baseline statistics on training set (excluding undecided/NaN fraud)
        labeled_data = data[data['is_fraud'].notna()].copy()
        global_fraud_mean = labeled_data['is_fraud'].mean()
        
        # Bayesian smoothed target encoding (m-estimate, m=10)
        m = 10.0
        p_stats = labeled_data.groupby('partner_id').agg(
            p_claims_count=('claim_id', 'count'),
            p_fraud_count=('is_fraud', 'sum'),
            p_under_2k_count=('is_under_2k', 'sum'),
            p_at_1995_count=('is_1995', 'sum'),
            p_uninspected_count=('uninspected_under_2k', 'sum')
        )
        p_stats['p_fraud_rate_bayes'] = (p_stats['p_fraud_count'] + m * global_fraud_mean) / (p_stats['p_claims_count'] + m)
        p_stats['p_under_2k_ratio'] = p_stats['p_under_2k_count'] / p_stats['p_claims_count'].clip(lower=1)
        p_stats['p_at_1995_ratio'] = p_stats['p_at_1995_count'] / p_stats['p_claims_count'].clip(lower=1)
        p_stats['p_uninspected_ratio'] = p_stats['p_uninspected_count'] / p_stats['p_claims_count'].clip(lower=1)
        
        partner_stats = {
            'global_mean': global_fraud_mean,
            'table': p_stats[['p_fraud_rate_bayes', 'p_under_2k_ratio', 'p_at_1995_ratio', 'p_uninspected_ratio', 'p_claims_count']]
        }
        
    # Map partner stats
    p_tbl = partner_stats['table']
    g_mean = partner_stats['global_mean']
    data = data.merge(p_tbl, on='partner_id', how='left')
    data['p_fraud_rate_bayes'] = data['p_fraud_rate_bayes'].fillna(g_mean)
    data['p_under_2k_ratio'] = data['p_under_2k_ratio'].fillna(0.0)
    data['p_at_1995_ratio'] = data['p_at_1995_ratio'].fillna(0.0)
    data['p_uninspected_ratio'] = data['p_uninspected_ratio'].fillna(0.0)
    data['p_claims_count'] = data['p_claims_count'].fillna(0.0)
    
    # 9. Categorical encodings
    data['partner_type_cat'] = data['partner_type'].astype('category').cat.codes
    data['family_cat'] = data['family'].astype('category').cat.codes
    data['city_cat'] = data['city'].astype('category').cat.codes
    data['fault_cat_code'] = data['fault_category'].astype('category').cat.codes
    
    return data, partner_stats
