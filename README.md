# Kestrel Home — Warranty Fraud Detection System

> 🌐 **Live Cloud Deployment & Investigation Desk:** [https://kestrel-warranty-fraud.onrender.com](https://kestrel-warranty-fraud.onrender.com)  
> 📖 **Interactive API Documentation (Swagger):** [https://kestrel-warranty-fraud.onrender.com/docs](https://kestrel-warranty-fraud.onrender.com/docs)  
> 📊 **Health Check Endpoint:** [https://kestrel-warranty-fraud.onrender.com/health](https://kestrel-warranty-fraud.onrender.com/health)  
> 🎥 **3-Minute Walkthrough Video:** [Google Drive Recording](https://drive.google.com/file/d/1_CGa9kTkivnh4RlTo_THvXOjYV3vPob3/view?usp=sharing)

A production-ready fraud detection pipeline for Kestrel Home Appliances' warranty claims. Built for the D2C Operations team to flag suspicious claims before payout, within the investigation desk's capacity of 40 reviews/month.

## Quick Start

### Prerequisites
- Python 3.9+
- No paid API keys required

### Installation & Run

```bash
# Clone the repository
git clone <repo-url>
cd kestrel-warranty-fraud

# Install dependencies
pip install xgboost scikit-learn pandas numpy fastapi uvicorn pydantic

# Start the service
python app.py
```

Open your browser to **http://127.0.0.1:8000** — the Investigation Desk UI loads automatically.

### Retrain the Model (Optional)

If you need to retrain from scratch using updated training data:

```bash
# Ensure data files are in data/ directory
cd src
python train.py
```

This regenerates `models/xgb_model.pkl`, `models/partner_stats.pkl`, `models/feature_cols.json`, and `predictions.csv`.

---

## Project Structure

```
kestrel-warranty-fraud/
├── app.py                  # FastAPI service + interactive UI (main entry point)
├── predictions.csv         # Final predictions for test_unlabelled.csv
├── memo.md                 # One-page memo to Ritu Deshpande
├── README.md               # This file
├── src/
│   ├── features.py         # Feature engineering pipeline (38 features)
│   └── train.py            # Model training, validation, prediction generation
├── models/
│   ├── xgb_model.pkl       # Trained XGBoost model (300 trees)
│   ├── partner_stats.pkl   # Bayesian partner fraud statistics
│   └── feature_cols.json   # Feature column order for inference
├── data/
│   ├── train.csv           # 12,029 historical claims (11,814 labeled)
│   ├── test_unlabelled.csv # 2,252 claims for scoring
│   ├── partners.csv        # 380 service partner reference data
│   └── products.csv        # 21 product SKUs with prices & warranty
├── email-thread.txt        # Client email thread (provided)
├── partners.csv            # Reference copy at root
├── products.csv            # Reference copy at root
└── sample_submission.csv   # Expected output format
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive Investigation Desk UI |
| `/health` | GET | Service health check |
| `/predict` | POST | Score a single claim (JSON -> fraud score + reasons) |
| `/sample_claims` | GET | Pre-loaded test scenarios |

### Example: Score a Claim

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "partner_id": "SP3310",
    "sku": "KH-AF-03",
    "product_serial": "KH335623749",
    "days_since_purchase": 268,
    "claim_amount_inr": 1995.0,
    "photo_attached": "N",
    "partner_inspected": "N",
    "claim_description": "blade jammed",
    "customer_prior_claims": 0,
    "source": "crm"
  }'
```

### Response

```json
{
  "claim_id": "WC711585",
  "fraud_score": 0.87,
  "risk_tier": "CRITICAL RISK",
  "recommendation": "HOLD FOR REVIEW (Priority Queue)",
  "action_code": "HOLD_REVIEW",
  "financials": {
    "claim_amount_inr": 1995.0,
    "goodwill_risk_inr": 380.0,
    "expected_prevented_fraud_inr": 1735.65,
    "net_roi_rupees": 1355.65
  },
  "reasons": [
    "Policy Loophole Gaming: Claim amount of Rs 1,995 is priced exactly Rs 5 below the Rs 2,000 inspection waiver limit.",
    "Evidence Gap: No fault photo attached and no inspection conducted by partner."
  ]
}
```

---

## Model Details

- **Algorithm:** XGBoost (gradient-boosted trees, 300 estimators, max_depth=4)
- **Features:** 38 engineered features across 9 categories
- **Training data:** 11,814 labeled claims (145 fraud, 11,669 genuine) - 15 months
- **Validation:** Temporal holdout (May-June 2026): ROC-AUC 0.86, PR-AUC 0.60
- **Cross-validation:** 5-fold stratified: OOF ROC-AUC 0.86, OOF PR-AUC 0.60

### Key Features
1. **Policy threshold gaming:** Claims at exactly Rs 1,995, proximity to Rs 2,000 cutoff
2. **Partner behavioral encoding:** Bayesian-smoothed historical fraud rate per outlet
3. **Adversarial text sanitization:** Regex-based stripping of prompt injection attempts
4. **Temporal awareness:** Pre/post May 2026 policy change detection
5. **Serial number anomalies:** Typography patterns in manually-entered serial numbers

---

## Key Finding: The Rs 2,000 Auto-Approval Loophole

The most actionable discovery: after the May 2026 policy change waiving inspection for claims under Rs 2,000, fraud rate doubled from 1.05% to 2.43%. 129 of 2,252 test claims (5.7%) are priced at exactly Rs 1,995 - systematic threshold gaming.

**Recommendation:** Close the loophole by requiring photo evidence on all claims regardless of amount. This process fix prevents fraud without needing ML.

---

## No Paid API Keys Required

The entire system runs locally using open-source libraries. No LLM APIs, no cloud ML services, no external dependencies beyond pip packages. Total operational cost per prediction: approximately Rs 0.00.
