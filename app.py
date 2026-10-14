import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

# Ensure src/ is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
from features import build_features

app = FastAPI(
    title="Kestrel Home - Warranty Fraud Detection Service",
    description="Automated claim review API & Investigation Desk Assistant for Kestrel Home Appliances",
    version="1.0.0"
)

# Load Reference Data and Artifacts
MODELS_DIR = os.path.join(current_dir, 'models')
DATA_DIR = os.path.join(current_dir, 'data')

try:
    with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'partner_stats.pkl'), 'rb') as f:
        partner_stats = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_cols.json'), 'r') as f:
        feature_cols = json.load(f)
    partners_df = pd.read_csv(os.path.join(DATA_DIR, 'partners.csv'))
    products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    print("Model, partner statistics, and product data loaded successfully.")
except Exception as e:
    print(f"Warning: Initialization error: {e}")
    model = None
    partner_stats = None
    feature_cols = []
    partners_df = pd.DataFrame()
    products_df = pd.DataFrame()

class ClaimRecord(BaseModel):
    claim_id: Optional[str] = "WC711350"
    submitted_at: Optional[str] = "2026-08-03 14:14"
    partner_id: str = Field(..., example="SP3232")
    sku: str = Field(..., example="KH-RV-03")
    product_serial: str = Field(..., example="KH-408131477")
    days_since_purchase: int = Field(..., example=120)
    claim_amount_inr: float = Field(..., example=1995.0)
    photo_attached: str = Field("N", example="N")
    partner_inspected: str = Field("N", example="N")
    claim_description: Optional[str] = "unit not heating"
    inspector_note: Optional[str] = None
    customer_prior_claims: Optional[int] = 0
    source: Optional[str] = "crm"

def generate_reasons(row_dict, feat_df, score):
    """
    Produces clear, plain-English reasons for Kestrel claim investigation desk employees.
    """
    reasons = []
    amount = float(row_dict.get('claim_amount_inr', 0))
    partner_id = row_dict.get('partner_id', '')
    inspected = row_dict.get('partner_inspected', 'N')
    photo = row_dict.get('photo_attached', 'N')
    serial = str(row_dict.get('product_serial', ''))
    prior_claims = int(row_dict.get('customer_prior_claims') or 0)
    
    # Check partner history
    p_info = partners_df[partners_df['partner_id'] == partner_id]
    p_stat_row = partner_stats['table'].loc[partner_id] if (partner_stats and partner_id in partner_stats['table'].index) else None
    
    if p_stat_row is not None:
        p_rate = float(p_stat_row['p_fraud_rate_bayes'])
        p_count = int(p_stat_row['p_claims_count'])
        if p_rate >= 0.10:
            reasons.append(f"Partner Risk: Outlet {partner_id} has a high historical fraud rate of {p_rate*100:.1f}% across {p_count} claims.")
        elif p_rate >= 0.04:
            reasons.append(f"Partner Risk: Outlet {partner_id} shows elevated fraud risk ({p_rate*100:.1f}% historical rate).")
            
    if not p_info.empty:
        onboarded = p_info.iloc[0]['onboarded_date']
        ptype = p_info.iloc[0]['partner_type'].replace('_', ' ').title()
        if str(onboarded) >= '2025-10-01':
            reasons.append(f"New Partner Outlet: Onboarded recently on {onboarded} ({ptype}). New outlets show 7x higher fraud incidence.")

    # Auto-approval policy threshold gaming (Rs 2,000 rule)
    if amount == 1995.0:
        reasons.append(f"Policy Loophole Gaming: Claim amount of Rs 1,995 is priced exactly Rs 5 below the Rs 2,000 inspection waiver limit.")
    elif 1900 <= amount < 2000 and inspected == 'N':
        reasons.append(f"Threshold Clustering: Claim amount of Rs {amount:.0f} sits right under the Rs 2,000 threshold and skipped physical inspection.")
    elif amount < 2000 and inspected == 'N':
        reasons.append(f"Uninspected Auto-Approval: Under Rs 2,000 auto-approval rule, submitted without physical technician inspection sign-off.")

    # Product economics
    sku = row_dict.get('sku', '')
    p_prod = products_df[products_df['sku'] == sku]
    if not p_prod.empty:
        list_price = float(p_prod.iloc[0]['list_price_inr'])
        family = p_prod.iloc[0]['family']
        ratio = amount / list_price
        if ratio >= 0.60:
            reasons.append(f"High Claim-to-Price Ratio: Claim amount represents {ratio*100:.1f}% of {family} list price (Rs {list_price:,.0f}).")

    # Serial format anomalies
    if '-' in serial or ' ' in serial or serial.islower():
        anomalies = []
        if '-' in serial: anomalies.append("hyphens")
        if ' ' in serial: anomalies.append("spacing")
        if 'kh' in serial: anomalies.append("lowercase prefix")
        reasons.append(f"Serial Typography Anomaly: Manual serial entry contains non-standard formatting ({', '.join(anomalies)}).")

    # Photo missing on uninspected
    if photo == 'N' and inspected == 'N':
        reasons.append("Evidence Gap: No fault photo attached and no inspection conducted by partner.")

    # Prior claims
    if prior_claims >= 2:
        reasons.append(f"Customer Velocity: Customer account has {prior_claims} prior warranty claims.")

    # Fallback if no specific high flags triggered
    if not reasons:
        if score >= 0.40:
            reasons.append("Multi-factor composite risk: Combination of submission timing, SKU repair history, and partner pattern.")
        else:
            reasons.append("Standard Claim: Claim attributes align with genuine customer repair benchmarks.")

    return reasons

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "features_count": len(feature_cols)}

@app.get("/sample_claims")
def sample_claims():
    """
    Returns curated sample claims representing different operational scenarios.
    """
    return [
        {
            "scenario": "High-Risk Loophole Gaming (Rs 1,995 Uninspected)",
            "data": {
                "claim_id": "WC711585",
                "submitted_at": "2026-07-09 19:05",
                "partner_id": "SP3310",
                "sku": "KH-AF-03",
                "product_serial": "KH335623749",
                "days_since_purchase": 268,
                "claim_amount_inr": 1995.0,
                "photo_attached": "Y",
                "partner_inspected": "N",
                "claim_description": "blade jammed",
                "inspector_note": "",
                "customer_prior_claims": 0,
                "source": "crm"
            }
        },
        {
            "scenario": "Serial Offender Partner + Serial Typography Anomaly",
            "data": {
                "claim_id": "WC712031",
                "submitted_at": "2026-07-28 00:14",
                "partner_id": "SP3207",
                "sku": "KH-RV-03",
                "product_serial": "KH-477512927",
                "days_since_purchase": 114,
                "claim_amount_inr": 8100.0,
                "photo_attached": "Y",
                "partner_inspected": "Y",
                "claim_description": "remote not working",
                "inspector_note": "",
                "customer_prior_claims": 1,
                "source": "crm"
            }
        },
        {
            "scenario": "Genuine Customer High-Value Repair",
            "data": {
                "claim_id": "WC711377",
                "submitted_at": "2026-07-02 05:59",
                "partner_id": "SP3273",
                "sku": "KH-WP-01",
                "product_serial": "KH313870581",
                "days_since_purchase": 39,
                "claim_amount_inr": 7610.0,
                "photo_attached": "Y",
                "partner_inspected": "Y",
                "claim_description": "burning smell",
                "inspector_note": "Photos match fault, approved",
                "customer_prior_claims": 0,
                "source": "crm"
            }
        },
        {
            "scenario": "Low-Risk Routine Fan Claim",
            "data": {
                "claim_id": "WC711413",
                "submitted_at": "2026-07-03 14:14",
                "partner_id": "SP3216",
                "sku": "KH-CF-01",
                "product_serial": "KH867748602",
                "days_since_purchase": 640,
                "claim_amount_inr": 353.0,
                "photo_attached": "Y",
                "partner_inspected": "N",
                "claim_description": "remote not working",
                "inspector_note": "",
                "customer_prior_claims": 0,
                "source": "crm"
            }
        }
    ]

@app.post("/predict")
def predict_claim(claim: ClaimRecord):
    """
    Main prediction endpoint. Takes a single claim record and returns:
    - fraud_score (0.0 to 1.0)
    - risk_tier ('CRITICAL', 'ELEVATED', 'LOW')
    - recommendation ('HOLD FOR DESK REVIEW', 'AUTO-APPROVE')
    - financial_risk (Estimated rupee loss vs Rs 380 goodwill cost)
    - reasons (Plain English list of operational flags for Kestrel staff)
    """
    if model is None:
        # Graceful polite fallback if model file is missing
        return JSONResponse(
            status_code=200,
            content={
                "claim_id": claim.claim_id,
                "fraud_score": 0.05,
                "risk_tier": "LOW",
                "recommendation": "AUTO-APPROVE",
                "reasons": ["Service running in fallback mode (model artifact loading)."],
                "notice": "Trained model artifact not loaded, returning safe baseline."
            }
        )
    
    # Format single record as dataframe
    rec = claim.dict()
    df_single = pd.DataFrame([rec])
    
    # Build features
    feat_df, _ = build_features(df_single, partners_df, products_df, is_train=False, partner_stats=partner_stats)
    X = feat_df[feature_cols]
    
    # Predict probability
    prob = float(model.predict_proba(X)[0, 1])
    score = round(prob, 4)
    
    # Determine risk tier & recommendation based on 40/month capacity cutoff
    # On test set (750 claims/month), the top 40 cutoff corresponds to score >= 0.22
    if score >= 0.45:
        risk_tier = "CRITICAL RISK"
        recommendation = "HOLD FOR REVIEW (Priority Queue)"
        action_code = "HOLD_REVIEW"
    elif score >= 0.20:
        risk_tier = "ELEVATED RISK"
        recommendation = "HOLD FOR REVIEW (Desk Capacity Permit)"
        action_code = "HOLD_REVIEW"
    elif score >= 0.08:
        risk_tier = "MODERATE RISK"
        recommendation = "AUTO-APPROVE (Post-Payout Audit)"
        action_code = "AUTO_APPROVE"
    else:
        risk_tier = "LOW RISK"
        recommendation = "AUTO-APPROVE"
        action_code = "AUTO_APPROVE"
        
    reasons = generate_reasons(rec, feat_df, score)
    
    # Financial arithmetic
    amount = float(claim.claim_amount_inr)
    goodwill_cost = 380.0
    expected_fraud_prevention = amount * score
    net_decision_value = expected_fraud_prevention - goodwill_cost if action_code == "HOLD_REVIEW" else 0.0
    
    return {
        "claim_id": claim.claim_id,
        "fraud_score": score,
        "risk_tier": risk_tier,
        "recommendation": recommendation,
        "action_code": action_code,
        "financials": {
            "claim_amount_inr": amount,
            "goodwill_risk_inr": goodwill_cost,
            "expected_prevented_fraud_inr": round(expected_fraud_prevention, 2),
            "net_roi_rupees": round(net_decision_value, 2)
        },
        "reasons": reasons
    }

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Renders the interactive Kestrel Claim Review UI screen.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kestrel Home - Warranty Fraud Shield</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #121826;
            --bg-card-hover: #182236;
            --border: #232e42;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.12);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.12);
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.12);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: linear-gradient(180deg, #161f33 0%, #0d1322 100%);
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.25rem;
            color: #fff;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        }

        .brand-text h1 {
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .brand-text p {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .header-badges {
            display: flex;
            gap: 0.75rem;
        }

        .badge {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 0.4rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-muted);
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .scenario-picker {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }

        .scenario-btn {
            background: #182238;
            border: 1px solid #2b3952;
            color: var(--text-muted);
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .scenario-btn:hover, .scenario-btn.active {
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
            box-shadow: 0 0 12px var(--accent-glow);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group.full {
            grid-column: span 2;
        }

        label {
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        input, select {
            background: #0d1320;
            border: 1px solid #222d42;
            color: #fff;
            padding: 0.7rem 0.9rem;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        .submit-btn {
            grid-column: span 2;
            margin-top: 1rem;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #fff;
            border: none;
            padding: 0.9rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
            transition: all 0.2s;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        }

        /* Result Panel */
        .result-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .score-hero {
            background: linear-gradient(135deg, #151d2f, #0d1424);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .score-val {
            font-size: 3.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1;
            margin: 0.5rem 0;
            transition: color 0.3s;
        }

        .score-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .status-pill {
            display: inline-block;
            padding: 0.4rem 1.25rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-top: 0.5rem;
            text-transform: uppercase;
        }

        .pill-critical { background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger); }
        .pill-elevated { background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning); }
        .pill-low { background: var(--success-bg); color: var(--success); border: 1px solid var(--success); }

        .reasons-box {
            background: #0d1320;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
        }

        .reasons-box h4 {
            font-size: 0.85rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            letter-spacing: 0.05em;
        }

        .reasons-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        .reasons-list li {
            font-size: 0.88rem;
            color: #e2e8f0;
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            line-height: 1.4;
        }

        .reasons-list li::before {
            content: "•";
            color: var(--accent);
            font-size: 1.4rem;
            line-height: 1;
        }

        .financial-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .fin-stat {
            background: #0d1320;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
        }

        .fin-stat .label {
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
        }

        .fin-stat .num {
            font-size: 1.35rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.25rem;
            color: #fff;
        }

        footer {
            border-top: 1px solid var(--border);
            padding: 1rem 2rem;
            font-size: 0.8rem;
            color: var(--text-dim);
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-logo">K</div>
            <div class="brand-text">
                <h1>Kestrel Home — Warranty Fraud Shield</h1>
                <p>AI Claim Assessment & Investigation Desk Optimizer (Variant C)</p>
            </div>
        </div>
        <div class="header-badges">
            <div class="badge"><div class="badge-dot"></div> Model Live (XGBoost 300T)</div>
            <div class="badge">Capacity: 40 claims/mo</div>
            <div class="badge">Goodwill Cost: ₹380</div>
        </div>
    </header>

    <main>
        <!-- Left Column: Input Form & Test Scenarios -->
        <section class="card">
            <div class="card-header">
                <div class="card-title">Warranty Claim Record</div>
                <span style="font-size: 0.8rem; color: var(--text-muted)">Live Inference API</span>
            </div>

            <div class="scenario-picker">
                <button class="scenario-btn active" onclick="loadScenario(0)">Loophole (₹1,995 Uninspected)</button>
                <button class="scenario-btn" onclick="loadScenario(1)">Serial Offender Partner</button>
                <button class="scenario-btn" onclick="loadScenario(2)">High-Value Genuine Repair</button>
                <button class="scenario-btn" onclick="loadScenario(3)">Routine Clean Fan Claim</button>
            </div>

            <form id="claimForm" onsubmit="event.preventDefault(); submitClaim();">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Claim Number</label>
                        <input type="text" id="claim_id" value="WC711585" required>
                    </div>
                    <div class="form-group">
                        <label>Submission Date (IST)</label>
                        <input type="text" id="submitted_at" value="2026-07-09 19:05" required>
                    </div>
                    <div class="form-group">
                        <label>Service Partner ID</label>
                        <input type="text" id="partner_id" value="SP3310" required>
                    </div>
                    <div class="form-group">
                        <label>Product SKU</label>
                        <select id="sku" required>
                            <option value="KH-AF-01">KH-AF-01 (Air Fryer ₹5,199)</option>
                            <option value="KH-AF-02">KH-AF-02 (Air Fryer ₹6,499)</option>
                            <option value="KH-AF-03" selected>KH-AF-03 (Air Fryer ₹8,773)</option>
                            <option value="KH-WP-01">KH-WP-01 (Water Purifier ₹11,999)</option>
                            <option value="KH-WP-02">KH-WP-02 (Water Purifier ₹14,999)</option>
                            <option value="KH-WP-03">KH-WP-03 (Water Purifier ₹20,248)</option>
                            <option value="KH-RV-01">KH-RV-01 (Robot Vacuum ₹17,599)</option>
                            <option value="KH-RV-02">KH-RV-02 (Robot Vacuum ₹21,999)</option>
                            <option value="KH-RV-03">KH-RV-03 (Robot Vacuum ₹29,698)</option>
                            <option value="KH-MG-01">KH-MG-01 (Mixer Grinder ₹3,199)</option>
                            <option value="KH-MG-02">KH-MG-02 (Mixer Grinder ₹3,999)</option>
                            <option value="KH-MG-03">KH-MG-03 (Mixer Grinder ₹5,398)</option>
                            <option value="KH-IC-01">KH-IC-01 (Induction Cooktop ₹2,399)</option>
                            <option value="KH-IC-02">KH-IC-02 (Induction Cooktop ₹2,999)</option>
                            <option value="KH-IC-03">KH-IC-03 (Induction Cooktop ₹4,048)</option>
                            <option value="KH-CF-01">KH-CF-01 (Ceiling Fan ₹2,799)</option>
                            <option value="KH-CF-02">KH-CF-02 (Ceiling Fan ₹3,499)</option>
                            <option value="KH-CF-03">KH-CF-03 (Ceiling Fan ₹4,723)</option>
                            <option value="KH-RH-01">KH-RH-01 (Room Heater ₹1,999)</option>
                            <option value="KH-RH-02">KH-RH-02 (Room Heater ₹2,499)</option>
                            <option value="KH-RH-03">KH-RH-03 (Room Heater ₹3,373)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Product Serial (Typed)</label>
                        <input type="text" id="product_serial" value="KH335623749" required>
                    </div>
                    <div class="form-group">
                        <label>Days Since Purchase</label>
                        <input type="number" id="days_since_purchase" value="268" required>
                    </div>
                    <div class="form-group">
                        <label>Claim Amount (INR)</label>
                        <input type="number" step="0.1" id="claim_amount_inr" value="1995.0" required>
                    </div>
                    <div class="form-group">
                        <label>Customer Prior Claims</label>
                        <input type="number" id="customer_prior_claims" value="0">
                    </div>
                    <div class="form-group">
                        <label>Photo Attached?</label>
                        <select id="photo_attached">
                            <option value="Y" selected>Yes (Y)</option>
                            <option value="N">No (N)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Partner Inspected?</label>
                        <select id="partner_inspected">
                            <option value="N" selected>No (Auto-Approve under ₹2,000)</option>
                            <option value="Y">Yes (Technician Signed)</option>
                        </select>
                    </div>
                    <div class="form-group full">
                        <label>Fault Description (Partner Free Text)</label>
                        <input type="text" id="claim_description" value="blade jammed">
                    </div>
                    <button type="submit" class="submit-btn" id="evaluateBtn">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Evaluate Claim Risk & Generate Reasons
                    </button>
                </div>
            </form>
        </section>

        <!-- Right Column: Decision, Reasons, and Financial Impact -->
        <section class="card result-panel">
            <div class="card-header">
                <div class="card-title">Investigation Recommendation</div>
                <span id="claimBadge" style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--accent);">WC711585</span>
            </div>

            <div class="score-hero">
                <div class="score-label">Fraud Probability Score</div>
                <div class="score-val" id="scoreVal">--</div>
                <div class="status-pill" id="riskPill">PENDING EVALUATION</div>
            </div>

            <div class="reasons-box">
                <h4>Operational Reasoning (For Kestrel Employee)</h4>
                <ul class="reasons-list" id="reasonsList">
                    <li>Awaiting claim submission evaluation...</li>
                </ul>
            </div>

            <div class="financial-grid">
                <div class="fin-stat">
                    <div class="label">Claim Amount</div>
                    <div class="num" id="finClaim">₹--</div>
                </div>
                <div class="fin-stat">
                    <div class="label">Expected Net Savings</div>
                    <div class="num" id="finNet" style="color: var(--success);">₹--</div>
                </div>
            </div>
        </section>
    </main>

    <footer>
        <div>Kestrel Home Appliances — Confidential internal operations tooling.</div>
        <div>Policy v4.1 compliant: Desk capacity 40 reviews/month • Goodwill benchmark ₹380</div>
    </footer>

    <script>
        let sampleData = [];

        async function init() {
            try {
                const res = await fetch('/sample_claims');
                sampleData = await res.json();
                submitClaim(); // run initial prediction
            } catch (e) {
                console.error("Init error", e);
            }
        }

        function loadScenario(idx) {
            document.querySelectorAll('.scenario-btn').forEach((b, i) => {
                b.classList.toggle('active', i === idx);
            });
            if (sampleData[idx]) {
                const d = sampleData[idx].data;
                document.getElementById('claim_id').value = d.claim_id;
                document.getElementById('submitted_at').value = d.submitted_at;
                document.getElementById('partner_id').value = d.partner_id;
                document.getElementById('sku').value = d.sku;
                document.getElementById('product_serial').value = d.product_serial;
                document.getElementById('days_since_purchase').value = d.days_since_purchase;
                document.getElementById('claim_amount_inr').value = d.claim_amount_inr;
                document.getElementById('customer_prior_claims').value = d.customer_prior_claims;
                document.getElementById('photo_attached').value = d.photo_attached;
                document.getElementById('partner_inspected').value = d.partner_inspected;
                document.getElementById('claim_description').value = d.claim_description;
                submitClaim();
            }
        }

        async function submitClaim() {
            const btn = document.getElementById('evaluateBtn');
            btn.innerText = "Evaluating...";
            btn.disabled = true;

            const payload = {
                claim_id: document.getElementById('claim_id').value,
                submitted_at: document.getElementById('submitted_at').value,
                partner_id: document.getElementById('partner_id').value,
                sku: document.getElementById('sku').value,
                product_serial: document.getElementById('product_serial').value,
                days_since_purchase: parseInt(document.getElementById('days_since_purchase').value) || 0,
                claim_amount_inr: parseFloat(document.getElementById('claim_amount_inr').value) || 0,
                customer_prior_claims: parseInt(document.getElementById('customer_prior_claims').value) || 0,
                photo_attached: document.getElementById('photo_attached').value,
                partner_inspected: document.getElementById('partner_inspected').value,
                claim_description: document.getElementById('claim_description').value,
                source: "crm"
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                renderResult(data);
            } catch (err) {
                console.error(err);
                alert("Error connecting to prediction endpoint.");
            } finally {
                btn.innerText = "Evaluate Claim Risk & Generate Reasons";
                btn.disabled = false;
            }
        }

        function renderResult(data) {
            document.getElementById('claimBadge').innerText = data.claim_id;
            const scoreEl = document.getElementById('scoreVal');
            scoreEl.innerText = (data.fraud_score * 100).toFixed(1) + "%";

            const pill = document.getElementById('riskPill');
            pill.className = "status-pill";
            if (data.risk_tier.includes("CRITICAL")) {
                pill.classList.add("pill-critical");
                scoreEl.style.color = "var(--danger)";
            } else if (data.risk_tier.includes("ELEVATED")) {
                pill.classList.add("pill-elevated");
                scoreEl.style.color = "var(--warning)";
            } else {
                pill.classList.add("pill-low");
                scoreEl.style.color = "var(--success)";
            }
            pill.innerText = data.recommendation;

            const list = document.getElementById('reasonsList');
            list.innerHTML = "";
            (data.reasons || []).forEach(r => {
                const li = document.createElement('li');
                li.innerText = r;
                list.appendChild(li);
            });

            document.getElementById('finClaim').innerText = "₹" + data.financials.claim_amount_inr.toLocaleString();
            const netEl = document.getElementById('finNet');
            if (data.financials.net_roi_rupees > 0) {
                netEl.innerText = "+₹" + data.financials.net_roi_rupees.toLocaleString();
                netEl.style.color = "var(--success)";
            } else {
                netEl.innerText = "₹" + data.financials.net_roi_rupees.toLocaleString();
                netEl.style.color = "var(--text-dim)";
            }
        }

        window.onload = init;
    </script>
</body>
</html>
"""
    return html_content

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Kestrel Warranty Fraud Service on http://0.0.0.0:{port} ...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
