"""
Standalone script to generate outputs/gold_annotation_sample.csv (Step 11).

Requires MongoDB to be running with amazon_reviews.reviews populated.
Uses outputs/food_safety_output.xlsx for DeBERTa columns (avoids re-running ML).
"""

import os
import pandas as pd
import numpy as np
from pymongo import MongoClient

MONGO_URI   = 'mongodb://localhost:27017/'
EXCEL_PATH  = 'outputs/food_safety_output.xlsx'
OUTPUT_DIR  = 'outputs'
SHEET_NAME  = 'MiniLM_DeBERTa_Confirmed'

# ── 1. MongoDB connection ──────────────────────────────────────────────
print("Connecting to MongoDB...")
COL = MongoClient(MONGO_URI)['amazon_reviews']['reviews']
print(f"  Total documents: {COL.count_documents({}):,}")

# ── 2. Build df_sem ──────────────────────────────────────────────────i don──
print("Building df_sem from MongoDB (no Text — lightweight)...")
cursor = COL.find(
    {'SemanticScore': {'$exists': True}},
    {'_id': 1, 'Score': 1, 'FoodSafetyFlag': 1,
     'SemanticScore': 1, 'SemanticFlagged': 1}
)
df_sem = pd.DataFrame(list(cursor))
df_sem['FoodSafetyFlag']  = df_sem['FoodSafetyFlag'].fillna(False).astype(bool)
df_sem['SemanticFlagged'] = df_sem['SemanticFlagged'].fillna(False).astype(bool)
print(f"  df_sem: {len(df_sem):,} rows")

# ── 3. Load DeBERTa results from Excel ────────────────────────────────
print(f"Loading DeBERTa results from {EXCEL_PATH}...")
confirmed_excel = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
print(f"  Excel confirmed rows: {len(confirmed_excel):,}")
print(f"  Columns: {list(confirmed_excel.columns)}")

# ── 4. Fetch Text for SemanticFlagged candidates ───────────────────────
print("Fetching Text from MongoDB for semantic candidates...")
sem_ids  = df_sem.loc[df_sem['SemanticFlagged'], '_id'].tolist()
id_text  = {
    d['_id']: d.get('Text', '')
    for d in COL.find({'_id': {'$in': sem_ids}}, {'_id': 1, 'Text': 1})
}
candidates_df = df_sem[df_sem['SemanticFlagged']].copy()
candidates_df['Text'] = candidates_df['_id'].map(id_text).fillna('')
print(f"  candidates_df: {len(candidates_df):,} rows")

# ── 5. Build Text → DeBERTa lookup from Excel ─────────────────────────
# Primary key: (Text, Score) — secondary fallback: Text only
text_score_map: dict = {}
text_only_map:  dict = {}

for _, row in confirmed_excel.iterrows():
    text_val   = str(row.get('Text', '')).strip()
    score_val  = row.get('Score', 0)
    entry = {
        'DebertaConfirmed':   True,
        'DebertaSafetyScore': float(row.get('DebertaSafetyScore', 0.0)),
        'DebertaCategory':    str(row.get('SafetyCategory', 'Unknown')),
        'FinalSafetyScore':   float(row.get('FinalSafetyScore', 0.0)),
        'ActionStatus':       str(row.get('ActionStatus', 'Not Evaluated')),
    }
    text_score_map[(text_val, score_val)] = entry
    if text_val not in text_only_map:
        text_only_map[text_val] = entry

_not_found = {
    'DebertaConfirmed':   False,
    'DebertaSafetyScore': 0.0,
    'DebertaCategory':    'Not_Evaluated',
    'FinalSafetyScore':   0.0,
    'ActionStatus':       'Not Evaluated',
}

def _lookup(row):
    key = (str(row['Text']).strip(), row.get('Score', 0))
    if key in text_score_map:
        return pd.Series(text_score_map[key])
    tk = str(row['Text']).strip()
    if tk in text_only_map:
        return pd.Series(text_only_map[tk])
    return pd.Series(_not_found)

deberta_data  = candidates_df.apply(_lookup, axis=1)
candidates_df = pd.concat([candidates_df, deberta_data], axis=1)

confirmed_df = candidates_df[candidates_df['DebertaConfirmed']].copy()
print(f"  confirmed_df: {len(confirmed_df):,} rows matched to Excel")

# ── 6. Recompute late-weighting (deterministic from Score) ────────────
def get_safety_weight(score: int) -> float:
    if score in (1, 2):   return 1.0
    elif score == 3:       return 0.8
    else:                  return 0.1

confirmed_df['Weight']         = confirmed_df['Score'].apply(get_safety_weight)
confirmed_df['FinalSafetyScore'] = (
    confirmed_df['DebertaSafetyScore'] * confirmed_df['Weight']
).round(4)

conditions = [
    confirmed_df['FinalSafetyScore'] >= 0.85,
    (confirmed_df['FinalSafetyScore'] >= 0.50) & (confirmed_df['FinalSafetyScore'] < 0.85),
]
choices = ['High Alert (Action Required)', 'Medium Alert (Manual Review)']
confirmed_df['ActionStatus'] = np.select(
    conditions, choices, default='Safe / False Positive'
)

# ── 7. Step 11: stratified sample ─────────────────────────────────────
print("\nBuilding stratified gold evaluation sample...")

# Use only base df_sem columns for sampling pools (DeBERTa added via merge below)
base_cols   = [c for c in ['_id', 'Score', 'FoodSafetyFlag', 'SemanticScore', 'SemanticFlagged']
               if c in df_sem.columns]
conf_sample = confirmed_df[base_cols].copy()

regex_pos    = df_sem[df_sem['FoodSafetyFlag']].copy()
semantic_pos = df_sem[df_sem['SemanticFlagged']].copy()
negatives    = df_sem[~df_sem['FoodSafetyFlag'] & ~df_sem['SemanticFlagged']].copy()

n_regex = min(100, len(regex_pos))
n_sem   = min(100, len(semantic_pos))
n_deb   = min(100, len(conf_sample))
n_neg   = min(100, len(negatives))

sample_eval = pd.concat([
    regex_pos.sample(n=n_regex, random_state=42),
    semantic_pos.sample(n=n_sem, random_state=42),
    conf_sample.sample(n=n_deb, random_state=42),
    negatives.sample(n=n_neg, random_state=42),
], ignore_index=False).drop_duplicates(subset=['_id']).copy()

# Fetch Text + Summary from MongoDB
print("Fetching Text + Summary from MongoDB for sampled reviews...")
eval_ids = sample_eval['_id'].tolist()
ts_map   = {
    d['_id']: (d.get('Text', ''), d.get('Summary', ''))
    for d in COL.find({'_id': {'$in': eval_ids}}, {'Text': 1, 'Summary': 1})
}
sample_eval['Text']    = sample_eval['_id'].map(lambda x: ts_map.get(x, ('', ''))[0])
sample_eval['Summary'] = sample_eval['_id'].map(lambda x: ts_map.get(x, ('', ''))[1])

# Merge DeBERTa columns from candidates_df
deberta_cols = candidates_df[['_id', 'DebertaConfirmed', 'DebertaSafetyScore', 'DebertaCategory']].copy()
sample_eval  = sample_eval.merge(deberta_cols, on='_id', how='left')
sample_eval['DebertaConfirmed']   = sample_eval['DebertaConfirmed'].fillna(False)
sample_eval['DebertaSafetyScore'] = sample_eval['DebertaSafetyScore'].fillna(0.0)
sample_eval['DebertaCategory']    = sample_eval['DebertaCategory'].fillna('Not_Evaluated')

# Merge late-weighting columns from confirmed_df
weight_cols  = confirmed_df[['_id', 'FinalSafetyScore', 'ActionStatus']].copy()
sample_eval  = sample_eval.merge(weight_cols, on='_id', how='left')
sample_eval['FinalSafetyScore'] = sample_eval['FinalSafetyScore'].fillna(0.0)
sample_eval['ActionStatus']     = sample_eval['ActionStatus'].fillna('Not Evaluated')

# ── 8. Export ──────────────────────────────────────────────────────────
keep_cols = ['_id', 'Score', 'Summary', 'Text',
             'FoodSafetyFlag', 'SemanticScore', 'SemanticFlagged',
             'DebertaConfirmed', 'DebertaSafetyScore', 'DebertaCategory',
             'FinalSafetyScore', 'ActionStatus']

gold_export = sample_eval[[c for c in keep_cols if c in sample_eval.columns]].copy()
gold_export['Human_Is_Hazard'] = ''
gold_export['Human_Category']  = ''
gold_export['Notes']           = ''

os.makedirs(OUTPUT_DIR, exist_ok=True)
gold_path = os.path.join(OUTPUT_DIR, 'gold_annotation_sample.csv')
gold_export.to_csv(gold_path, index=False)

print(f"\nGold annotation file saved: {gold_path}")
print(f"Total rows to annotate: {len(gold_export)}")
print(f"\nBreakdown (before dedup):")
print(f"  Regex positive:    {n_regex}")
print(f"  Semantic positive: {n_sem}")
print(f"  DeBERTa confirmed: {n_deb}")
print(f"  Negatives:         {n_neg}")
print(f"  After dedup:       {len(gold_export)}")
print("\nNext: open the CSV in Excel, fill Human_Is_Hazard + Human_Category, then run Step 12.")
