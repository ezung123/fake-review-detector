import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ml.src.data.adapters.ott import load_ott
from ml.src.data.adapters.salminen import load_salminen
from ml.src.data.adapters.raid import load_raid

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SEED = 42

def run_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Load Raw Datasets
    # -------------------------------------------------------------
    print("Loading raw datasets...")
    ott_raw_path = RAW_DIR / "ott" / "deceptive.csv"
    sal_raw_path = RAW_DIR / "salminen" / "fake_reviews_dataset.csv"
    raid_raw_path = RAW_DIR / "raid" / "raid_reviews.csv"
    
    # Initial load for raw row count verification
    ott_initial = pd.read_csv(ott_raw_path)
    sal_initial = pd.read_csv(sal_raw_path)
    raid_initial = pd.read_csv(raid_raw_path)
    raid_filtered_initial = raid_initial[
        (raid_initial['domain'] == 'reviews') & (raid_initial['attack'] == 'none')
    ]
    
    # -------------------------------------------------------------
    # 2. Adapters & Deduplication
    # -------------------------------------------------------------
    # Ott
    ott_df = load_ott(ott_raw_path)
    ott_raw_count = len(ott_initial)
    ott_df = ott_df.drop_duplicates(subset=['text']).copy()
    ott_dedup_count = len(ott_df)
    ott_removed = ott_raw_count - ott_dedup_count
    
    # Salminen
    sal_df = load_salminen(sal_raw_path)
    sal_raw_count = len(sal_initial)
    sal_df = sal_df.drop_duplicates(subset=['text']).copy()
    sal_dedup_count = len(sal_df)
    sal_removed = sal_raw_count - sal_dedup_count
    
    # RAID: Deduplicate on filtered reviews before sampling
    raid_raw_count = len(raid_filtered_initial)
    # Deduplicate raw raid reviews text first
    raid_filtered_dedup = raid_filtered_initial.copy()
    raid_filtered_dedup['norm_text'] = raid_filtered_dedup['generation'].astype(str).str.strip().apply(lambda s: ' '.join(s.split()))
    raid_filtered_dedup = raid_filtered_dedup.drop_duplicates(subset=['norm_text']).copy()
    raid_dedup_count = len(raid_filtered_dedup)
    raid_removed = raid_raw_count - raid_dedup_count
    
    # Sample AI rows to 2x human count evenly across 11 models
    human_raid = raid_filtered_dedup[raid_filtered_dedup['model'] == 'human'].copy()
    ai_raid = raid_filtered_dedup[raid_filtered_dedup['model'] != 'human'].copy()
    
    n_human_raid = len(human_raid) # 943
    target_ai_raid = n_human_raid * 2 # 1886
    models = sorted(ai_raid['model'].unique())
    base_count = target_ai_raid // len(models) # 171
    remainder = target_ai_raid % len(models) # 5
    
    sampled_ai_parts = []
    for idx, model_name in enumerate(models):
        m_count = base_count + (1 if idx < remainder else 0)
        subset = ai_raid[ai_raid['model'] == model_name]
        sampled_subset = subset.sample(n=m_count, random_state=SEED)
        sampled_ai_parts.append(sampled_subset)
        
    sampled_ai_raid = pd.concat(sampled_ai_parts, ignore_index=True)
    raid_sampled_combined = pd.concat([human_raid, sampled_ai_raid], ignore_index=True)
    
    raid_df = pd.DataFrame({
        'original_id': raid_sampled_combined['id'].astype(str),
        'source_id': raid_sampled_combined['source_id'].astype(str),
        'text': raid_sampled_combined['norm_text'],
        'source_dataset': 'raid',
        'authorship_label': np.where(raid_sampled_combined['model'] == 'human', 'human', 'ai'),
        'deception_label': 'unknown',
        'domain': 'movie',
    })
    
    # -------------------------------------------------------------
    # 3. Stratified & Group-Aware Splits (Seed: 42)
    # -------------------------------------------------------------
    # A. RAID: Group-aware split on source_id (80 / 10 / 10)
    unique_sources = human_raid['source_id'].astype(str).unique()
    rng = np.random.RandomState(SEED)
    shuffled_sources = rng.permutation(unique_sources)
    
    n_sources = len(shuffled_sources)
    n_train_s = int(0.80 * n_sources) # 754
    n_val_s = int(0.10 * n_sources)   # 94
    # remainder to test: 95
    
    train_source_set = set(shuffled_sources[:n_train_s])
    val_source_set = set(shuffled_sources[n_train_s:n_train_s + n_val_s])
    test_source_set = set(shuffled_sources[n_train_s + n_val_s:])
    
    def assign_raid_split(s_id):
        if s_id in train_source_set:
            return 'train'
        elif s_id in val_source_set:
            return 'val'
        return 'test'
        
    raid_df['split'] = raid_df['source_id'].apply(assign_raid_split)
    raid_df['deception_cv_fold'] = -1
    
    # B. Salminen: Stratified split on authorship_label (80 / 10 / 10)
    # 80% train, 20% temp -> 10% val, 10% test
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, temp_idx = next(sss_outer.split(sal_df, sal_df['authorship_label']))
    
    sal_train = sal_df.iloc[train_idx].copy()
    sal_temp = sal_df.iloc[temp_idx].copy()
    
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel_idx, test_rel_idx = next(sss_inner.split(sal_temp, sal_temp['authorship_label']))
    
    sal_val = sal_temp.iloc[val_rel_idx].copy()
    sal_test = sal_temp.iloc[test_rel_idx].copy()
    
    sal_train['split'] = 'train'
    sal_val['split'] = 'val'
    sal_test['split'] = 'test'
    
    sal_df = pd.concat([sal_train, sal_val, sal_test], ignore_index=True)
    sal_df['deception_cv_fold'] = -1
    
    # C. Ott: Exclude from authorship training, held-out test + 5-fold CV for deception
    ott_df['split'] = 'test' # Held-out test for authorship evaluation
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ott_df['deception_cv_fold'] = -1
    for fold_num, (_, fold_val_idx) in enumerate(skf.split(ott_df, ott_df['deception_label'])):
        ott_df.iloc[fold_val_idx, ott_df.columns.get_loc('deception_cv_fold')] = fold_num
        
    # -------------------------------------------------------------
    # 4. Combine into Unified Dataset
    # -------------------------------------------------------------
    ordered_cols = [
        'row_id', 'original_id', 'source_id', 'text', 'source_dataset',
        'authorship_label', 'deception_label', 'domain', 'split', 'deception_cv_fold'
    ]
    
    sal_df['row_id'] = sal_df['original_id']
    raid_df['row_id'] = raid_df['original_id']
    ott_df['row_id'] = ott_df['original_id']
    
    unified_df = pd.concat([
        sal_df[ordered_cols],
        raid_df[ordered_cols],
        ott_df[ordered_cols]
    ], ignore_index=True)
    
    # -------------------------------------------------------------
    # 5. Export Datasets
    # -------------------------------------------------------------
    unified_csv = PROCESSED_DIR / "unified_dataset.csv"
    unified_parquet = PROCESSED_DIR / "unified_dataset.parquet"
    train_csv = PROCESSED_DIR / "train.csv"
    val_csv = PROCESSED_DIR / "val.csv"
    test_csv = PROCESSED_DIR / "test.csv"
    ott_cv_csv = PROCESSED_DIR / "ott_deception_5fold.csv"
    
    unified_df.to_csv(unified_csv, index=False)
    unified_df.to_parquet(unified_parquet, index=False)
    
    unified_df[unified_df['split'] == 'train'].to_csv(train_csv, index=False)
    unified_df[unified_df['split'] == 'val'].to_csv(val_csv, index=False)
    unified_df[unified_df['split'] == 'test'].to_csv(test_csv, index=False)
    ott_df[ordered_cols].to_csv(ott_cv_csv, index=False)
    
    # -------------------------------------------------------------
    # 6. Reporting
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("DEDUPLICATION SUMMARY:")
    print("="*80)
    print(f"Ott et al.:      {ott_raw_count:,} raw rows -> {ott_dedup_count:,} unique ({ott_removed} duplicates removed)")
    print(f"Salminen et al.: {sal_raw_count:,} raw rows -> {sal_dedup_count:,} unique ({sal_removed} duplicates removed)")
    print(f"RAID (reviews):  {raid_raw_count:,} raw rows -> {raid_dedup_count:,} unique ({raid_removed} duplicates removed)")
    print(f"Total removed:   {ott_removed + sal_removed + raid_removed} duplicate rows")
    
    print("\n" + "="*80)
    print("DATASET BREAKDOWN BY source_dataset x authorship_label x deception_label x split:")
    print("="*80)
    count_table = pd.crosstab(
        index=[unified_df['source_dataset'], unified_df['authorship_label'], unified_df['deception_label']],
        columns=unified_df['split'],
        margins=True,
        margins_name='Total'
    )
    print(count_table.to_string())
    
    print("\n" + "="*80)
    print("OTT ET AL. 5-FOLD STRATIFIED CV BREAKDOWN (FOR DECEPTION TASK):")
    print("="*80)
    ott_cv_table = pd.crosstab(
        index=ott_df['deception_cv_fold'],
        columns=ott_df['deception_label'],
        margins=True,
        margins_name='Total'
    )
    ott_cv_table.index.name = "CV Fold"
    print(ott_cv_table.to_string())
    
    print("\n" + "="*80)
    print("VERIFICATION CHECKS:")
    print("="*80)
    # Verify no source_id leaks in RAID
    raid_train_s = set(unified_df[(unified_df['source_dataset'] == 'raid') & (unified_df['split'] == 'train')]['source_id'])
    raid_val_s = set(unified_df[(unified_df['source_dataset'] == 'raid') & (unified_df['split'] == 'val')]['source_id'])
    raid_test_s = set(unified_df[(unified_df['source_dataset'] == 'raid') & (unified_df['split'] == 'test')]['source_id'])
    
    assert len(raid_train_s & raid_val_s) == 0, "RAID data leakage detected between train and val!"
    assert len(raid_train_s & raid_test_s) == 0, "RAID data leakage detected between train and test!"
    assert len(raid_val_s & raid_test_s) == 0, "RAID data leakage detected between val and test!"
    print("✓ RAID group-aware split verified: 0 source_id overlap across train, val, and test.")
    
    # Verify Ott has 0 rows in train and val
    ott_in_train = len(unified_df[(unified_df['source_dataset'] == 'ott') & (unified_df['split'] == 'train')])
    ott_in_val = len(unified_df[(unified_df['source_dataset'] == 'ott') & (unified_df['split'] == 'val')])
    assert ott_in_train == 0 and ott_in_val == 0, "Ott leaked into authorship training!"
    print(f"✓ Ott excluded from authorship training: {ott_in_train} in train, {ott_in_val} in val.")
    print(f"✓ All processed datasets successfully written to {PROCESSED_DIR}")

if __name__ == "__main__":
    run_pipeline()

