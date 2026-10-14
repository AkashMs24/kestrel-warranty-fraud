import os
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from features import build_features

def evaluate_financials(y_true, y_scores, claim_amounts, top_n=120, goodwill_cost=380.0):
    """
    Computes business financial impact for top-N flagged claims (40 claims/month * 3 months = 120 claims).
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'score': y_scores,
        'amount': claim_amounts
    }).sort_values('score', ascending=False).reset_index(drop=True)
    
    top_claims = df.iloc[:top_n]
    true_positives = (top_claims['y_true'] == 1).sum()
    false_positives = (top_claims['y_true'] == 0).sum()
    precision_at_n = true_positives / top_n
    total_fraud_available = (df['y_true'] == 1).sum()
    recall_at_n = true_positives / total_fraud_available if total_fraud_available > 0 else 0
    
    fraud_rupees_prevented = top_claims[top_claims['y_true'] == 1]['amount'].sum()
    goodwill_payout_cost = false_positives * goodwill_cost
    net_rupees_saved = fraud_rupees_prevented - goodwill_payout_cost
    rupees_saved_per_check = net_rupees_saved / top_n
    
    return {
        'top_n_checked': top_n,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'precision_at_n': float(precision_at_n),
        'recall_at_n': float(recall_at_n),
        'fraud_rupees_prevented': float(fraud_rupees_prevented),
        'goodwill_cost': float(goodwill_payout_cost),
        'net_rupees_saved': float(net_rupees_saved),
        'rupees_saved_per_check': float(rupees_saved_per_check)
    }

def main():
    project_dir = r'C:\Users\bumik\.gemini\antigravity-ide\scratch\kestrel-warranty-fraud'
    train_raw = pd.read_csv(os.path.join(project_dir, 'data', 'train.csv'))
    partners = pd.read_csv(os.path.join(project_dir, 'data', 'partners.csv'))
    products = pd.read_csv(os.path.join(project_dir, 'data', 'products.csv'))
    test_raw = pd.read_csv(os.path.join(project_dir, 'data', 'test_unlabelled.csv'))
    
    # Exclude undecided / open investigation claims (NaN) from supervised training labels
    train_labeled = train_raw[train_raw['is_fraud'].notna()].copy()
    train_labeled['is_fraud'] = train_labeled['is_fraud'].astype(int)
    
    print(f'Training samples (labeled): {len(train_labeled)} (Fraud: {train_labeled["is_fraud"].sum()})')
    
    # Feature engineering
    train_feat, partner_stats = build_features(train_labeled, partners, products, is_train=True)
    test_feat, _ = build_features(test_raw, partners, products, is_train=False, partner_stats=partner_stats)
    
    feature_cols = [
        'claim_amount_inr', 'is_under_2k', 'is_1995', 'is_near_2k', 'dist_to_2k',
        'claim_ratio', 'list_price_inr', 'warranty_months', 'days_since_purchase',
        'warranty_consumed_pct', 'is_late_warranty', 'is_early_warranty',
        'photo_attached_flag', 'partner_inspected_flag', 'uninspected_under_2k',
        'has_inspector_note', 'customer_prior_claims_num', 'source_crm',
        'partner_age_days', 'is_new_partner', 'is_post_may_2026',
        'sub_hour', 'sub_dow', 'is_weekend', 'is_night',
        'serial_has_space', 'serial_has_dash', 'serial_is_lower', 'serial_len',
        'partner_type_cat', 'family_cat', 'city_cat', 'fault_cat_code',
        'p_fraud_rate_bayes', 'p_under_2k_ratio', 'p_at_1995_ratio',
        'p_uninspected_ratio', 'p_claims_count'
    ]
    
    print(f'Features count: {len(feature_cols)}')
    
    # Temporal validation split: Train on < 2026-05-01, Validate on May-June 2026 (post-May policy change!)
    val_mask = train_feat['submitted_at'] >= '2026-05-01'
    X_train_val = train_feat.loc[~val_mask, feature_cols]
    y_train_val = train_feat.loc[~val_mask, 'is_fraud']
    X_val = train_feat.loc[val_mask, feature_cols]
    y_val = train_feat.loc[val_mask, 'is_fraud']
    val_amounts = train_feat.loc[val_mask, 'claim_amount_inr'].values
    
    print(f'Temporal Split: Train before May 2026 ({len(X_train_val)} rows, {y_train_val.sum()} fraud), Val May-June 2026 ({len(X_val)} rows, {y_val.sum()} fraud)')
    
    # Scale positive weight due to 1:80 class imbalance
    scale_pos = (len(y_train_val) - y_train_val.sum()) / y_train_val.sum()
    
    val_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='aucpr'
    )
    val_model.fit(X_train_val, y_train_val)
    val_preds = val_model.predict_proba(X_val)[:, 1]
    
    roc_val = roc_auc_score(y_val, val_preds)
    pr_val = average_precision_score(y_val, val_preds)
    print(f'Temporal Validation ROC-AUC: {roc_val:.4f}, PR-AUC: {pr_val:.4f}')
    
    # Validation period is 2 months (May and June 2026), capacity is 40/month = 80 claims
    fin_val = evaluate_financials(y_val.values, val_preds, val_amounts, top_n=80, goodwill_cost=380.0)
    print('\n=== TEMPORAL VALIDATION FINANCIALS (May-June 2026: 80 reviews) ===')
    for k, v in fin_val.items():
        print(f'{k}: {v}')
        
    # Full Model: Train on all labeled data (15 months)
    X_full = train_feat[feature_cols]
    y_full = train_feat['is_fraud']
    scale_pos_full = (len(y_full) - y_full.sum()) / y_full.sum()
    
    full_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_full,
        random_state=42,
        eval_metric='aucpr'
    )
    full_model.fit(X_full, y_full)
    
    # Save model and artifacts
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(full_model, f)
    with open(os.path.join(models_dir, 'partner_stats.pkl'), 'wb') as f:
        pickle.dump(partner_stats, f)
    with open(os.path.join(models_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
        
    print('\nSuccessfully trained and saved model artifacts!')
    
    # Predict on test set
    X_test = test_feat[feature_cols]
    test_preds = full_model.predict_proba(X_test)[:, 1]
    
    # Format predictions.csv
    pred_df = pd.DataFrame({
        'claim_id': test_feat['claim_id'],
        'score': np.round(test_preds, 4)
    })
    
    # Save predictions.csv to repository root and data/
    pred_path_root = os.path.join(project_dir, 'predictions.csv')
    pred_path_data = os.path.join(project_dir, 'data', 'predictions.csv')
    pred_df.to_csv(pred_path_root, index=False)
    pred_df.to_csv(pred_path_data, index=False)
    
    print(f'Generated predictions.csv with {len(pred_df)} rows.')
    print('Predictions preview:')
    print(pred_df.head(10))
    print('Score summary stats:')
    print(pred_df['score'].describe())
    
    # Test financials expectation (3 months, 40/month = 120 claims)
    top_120_test = pred_df.sort_values('score', ascending=False).head(120)
    print('\nTop 120 test scores range:', top_120_test['score'].min(), 'to', top_120_test['score'].max())
    print('Top 120 claims test amounts sum:', test_feat.loc[top_120_test.index, 'claim_amount_inr'].sum())

if __name__ == '__main__':
    main()
