"""
Step 11 equivalent — Stratified Gold Evaluation Sample
=======================================================
Stratified by detection stage (4 strata x 100 samples = up to 400 rows):
  Stratum 1 — Regex positives    : FoodSafetyFlag == True
  Stratum 2 — Semantic positives : SemanticFlagged == True
  Stratum 3 — DeBERTa confirmed  : DebertaConfirmed == True
  Stratum 4 — Negatives          : FoodSafetyFlag == False AND SemanticFlagged == False

Sources (no MongoDB required):
  - Reviews_withURL.csv           -> Id, Score, Summary, Text for all 568k reviews
  - outputs/food_safety_output.xlsx -> ML pipeline outcomes (flags, scores, categories)
"""

import os
import pandas as pd
import numpy as np

CSV_PATH    = 'Reviews_withURL.csv'
EXCEL_PATH  = 'outputs/food_safety_output.xlsx'
OUTPUT_PATH = 'outputs/gold_annotation_sample.csv'
N_PER_STRATUM = 100
RANDOM_STATE  = 42

# ── 1. Load all reviews ────────────────────────────────────────────────
print("Loading Reviews_withURL.csv...")
reviews = pd.read_csv(CSV_PATH, usecols=['Id', 'Score', 'Summary', 'Text'])
reviews['Text'] = reviews['Text'].fillna('').astype(str).str.strip()
print(f"  {len(reviews):,} reviews loaded")

# ── 2. Load ML pipeline outcomes from Excel ────────────────────────────
print("Loading food_safety_output.xlsx...")
regex_xl    = pd.read_excel(EXCEL_PATH, sheet_name='Regex_Flagged')
implicit_xl = pd.read_excel(EXCEL_PATH, sheet_name='MiniLM_Implicit_Only')
deberta_xl  = pd.read_excel(EXCEL_PATH, sheet_name='MiniLM_DeBERTa_Confirmed')

for df in [regex_xl, implicit_xl, deberta_xl]:
    df['Text'] = df['Text'].fillna('').astype(str).str.strip()

print(f"  Regex_Flagged:            {len(regex_xl):,}")
print(f"  MiniLM_Implicit_Only:     {len(implicit_xl):,}")
print(f"  MiniLM_DeBERTa_Confirmed: {len(deberta_xl):,}")

# ── 3. Tag reviews using Text as join key ──────────────────────────────
print("Tagging reviews with pipeline flags...")

# FoodSafetyFlag and SemanticFlagged
regex_text_set    = set(regex_xl['Text'])
semantic_text_set = set(implicit_xl['Text']) | set(deberta_xl['Text'])

reviews['FoodSafetyFlag']  = reviews['Text'].isin(regex_text_set)
reviews['SemanticFlagged'] = reviews['Text'].isin(semantic_text_set)

# SemanticScore — deberta sheet takes priority over implicit
sem_score_map = (
    dict(zip(implicit_xl['Text'], implicit_xl['SemanticScore'])) |
    dict(zip(deberta_xl['Text'],  deberta_xl['SemanticScore']))
)
reviews['SemanticScore'] = reviews['Text'].map(sem_score_map).fillna(0.0)

# DeBERTa columns (only for confirmed rows)
deberta_map = {}
for _, row in deberta_xl.iterrows():
    t = row['Text']
    if t not in deberta_map:
        deberta_map[t] = {
            'DebertaConfirmed':   True,
            'DebertaSafetyScore': float(row.get('DebertaSafetyScore', 0.0)),
            'DebertaCategory':    str(row.get('SafetyCategory', 'Unknown')),
            'FinalSafetyScore':   float(row.get('FinalSafetyScore', 0.0)),
            'ActionStatus':       str(row.get('ActionStatus', 'Not Evaluated')).encode('ascii', 'ignore').decode().strip(),
        }

_default = {
    'DebertaConfirmed': False, 'DebertaSafetyScore': 0.0,
    'DebertaCategory': 'Not_Evaluated', 'FinalSafetyScore': 0.0,
    'ActionStatus': 'Not Evaluated',
}
for col in _default:
    reviews[col] = reviews['Text'].map(lambda t, c=col: deberta_map.get(t, _default)[c])

# ── 4. Build strata ────────────────────────────────────────────────────
regex_pos    = reviews[reviews['FoodSafetyFlag']].copy()
semantic_pos = reviews[reviews['SemanticFlagged']].copy()
deberta_pos  = reviews[reviews['DebertaConfirmed']].copy()
negatives    = reviews[~reviews['FoodSafetyFlag'] & ~reviews['SemanticFlagged']].copy()

n_regex = min(N_PER_STRATUM, len(regex_pos))
n_sem   = min(N_PER_STRATUM, len(semantic_pos))
n_deb   = min(N_PER_STRATUM, len(deberta_pos))
n_neg   = min(N_PER_STRATUM, len(negatives))

print(f"\nStrata sizes:")
print(f"  Stratum 1 - Regex positives:    {len(regex_pos):,}  (sampling {n_regex})")
print(f"  Stratum 2 - Semantic positives: {len(semantic_pos):,}  (sampling {n_sem})")
print(f"  Stratum 3 - DeBERTa confirmed:  {len(deberta_pos):,}  (sampling {n_deb})")
print(f"  Stratum 4 - Negatives:          {len(negatives):,}  (sampling {n_neg})")

# ── 5. Stratified sample + dedup ──────────────────────────────────────
sample_eval = pd.concat([
    regex_pos.sample(n=n_regex, random_state=RANDOM_STATE),
    semantic_pos.sample(n=n_sem, random_state=RANDOM_STATE),
    deberta_pos.sample(n=n_deb, random_state=RANDOM_STATE),
    negatives.sample(n=n_neg,   random_state=RANDOM_STATE),
], ignore_index=True).drop_duplicates(subset=['Id']).copy()

# ── 6. Select columns and add annotation fields ────────────────────────
keep_cols = [
    'Id', 'Score', 'Summary', 'Text',
    'FoodSafetyFlag', 'SemanticScore', 'SemanticFlagged',
    'DebertaConfirmed', 'DebertaSafetyScore', 'DebertaCategory',
    'FinalSafetyScore', 'ActionStatus',
]
gold = sample_eval[[c for c in keep_cols if c in sample_eval.columns]].copy()
gold['Human_Is_Hazard'] = ''   # 1 = genuine hazard, 0 = not
gold['Human_Category']  = ''   # Illness / Contamination / Spoilage / Allergen / Quality_Defect / Safe
gold['Notes']           = ''

# ── 7. Save ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
gold.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')


print(f"\nSaved: {OUTPUT_PATH}")
print(f"Total rows: {len(gold)}")
print(f"\nBreakdown (after dedup):")
print(f"  Regex positive:    {n_regex}")
print(f"  Semantic positive: {n_sem}")
print(f"  DeBERTa confirmed: {n_deb}")
print(f"  Negatives:         {n_neg}")
print(f"  Final total:       {len(gold)}")
print("\nNext: open CSV in Excel, fill Human_Is_Hazard + Human_Category, then run Step 12.")
