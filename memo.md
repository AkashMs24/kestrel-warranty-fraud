# Memo to Ritu Deshpande — Warranty Fraud Model Results

**From:** Data Team
**To:** Ritu Deshpande, Head of D2C Operations
**Date:** October 2026
**Re:** Warranty fraud flagging — what the model found and what to do next

---

## The Short Version

We built a fraud-scoring model on your 15 months of claim data. It works — but the biggest thing we found isn't about the model. It's about a policy loophole.

---

## What We Found

**Your fraud rate is 1.2% overall — but it doubled to 2.4% after the May policy change.**

The May 2026 update waived partner inspection for claims under Rs 2,000. Since then, 129 claims in the latest quarter alone are priced at exactly Rs 1,995 — five rupees under the threshold. These are not coincidences. Certain outlets are pricing claims just below the inspection cutoff to get auto-approved without a technician visit.

**Your instinct about newer partners was partially right** — new outlets (onboarded after October 2025) show 7x the fraud incidence. But the root cause isn't that they're new. It's that the under-Rs 2,000 rule gives them a shortcut they're exploiting.

---

## The Numbers

| Metric | Value |
|--------|-------|
| Claims reviewed by model | 11,814 (training), 2,252 (scoring) |
| Confirmed fraud in training | 145 cases (1.2%) |
| Model accuracy (ROC-AUC) | 0.86 on held-out May-June 2026 data |
| Top 40 flagged claims | All score above 91% fraud probability |
| Estimated fraud value in top-40 flags | Rs 63,000+ per month |
| Monthly desk capacity (Farhan's team) | 40 reviews |
| Goodwill cost per false positive | Rs 380 |
| Projected annual savings | Rs 4-6 lakh |

---

## The "97% Accuracy" Question

Your board asked for 97% accuracy. Here's the honest answer: a model that approves every single claim — never catching any fraud — would score 98.8% accuracy, because 98.8% of claims are genuine.

Accuracy is the wrong yardstick when fraud is rare. What matters is: **of the 40 claims Farhan's team reviews each month, how many are actually fraud?** Our model puts the highest-probability cases at the top of that list. That's where the rupees are saved.

---

## What to Do Next Week

1. **Close the Rs 2,000 loophole.** Require photo evidence on all warranty claims, regardless of amount. This single policy change eliminates the easiest fraud vector immediately — no model needed.

2. **Deploy the scoring model.** The system is ready: `python app.py` on any machine. It scores claims in under 3 milliseconds each and costs nothing to run (no paid APIs). Have Meenal's desk team start with the top-40 flag list each month.

3. **Watch the Rs 1,995 pattern.** If it drops after the policy change, the loophole was the driver. If it shifts to Rs 1,990 or Rs 1,985, the gaming is adaptive and you need the model as a permanent backstop.

4. **Don't paint all new partners with one brush** (Meenal's concern is valid). The model scores each claim individually — a new partner with clean patterns gets low scores. But do add a 90-day probation period with mandatory inspection for new outlets.

---

## The Bottom Line

The model saves Rs 4-6 lakh/year within your existing 40-review desk capacity. But the bigger win is the policy fix: close the under-Rs 2,000 auto-approval loophole, and you eliminate the fraud pattern that's been growing since May.

The system runs on any laptop, costs nothing, and starts with one command. It's ready when you are.
